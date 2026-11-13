// sherpa-onnx/csrc/offline-audio8-asr-model.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/csrc/offline-audio8-asr-model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineAudio8AsrModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR, "audio8-asr"),
        sess_opts_encoder_(GetSessionOptions(config)),
        sess_opts_decoder_(GetSessionOptions(config)),
        allocator_(),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    const auto &c = config_.audio8_asr;

    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(c.encoder), sess_opts_encoder_);
    InitEncoder();

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(c.decoder), sess_opts_decoder_);
    InitDecoder();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR, "audio8-asr"),
        sess_opts_encoder_(GetSessionOptions(config)),
        sess_opts_decoder_(GetSessionOptions(config)),
        allocator_(),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    const auto &c = config_.audio8_asr;

    {
      auto buf = ReadFile(mgr, c.encoder);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read audio8_asr.encoder: %s",
                         c.encoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_encoder_);
      InitEncoder();
    }

    {
      auto buf = ReadFile(mgr, c.decoder);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read audio8_asr.decoder: %s",
                         c.decoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      decoder_sess_ = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_decoder_);
      InitDecoder();
    }
  }

  Ort::Value ForwardEncoder(Ort::Value audios, Ort::Value num_rows) const {
    std::array<Ort::Value, 2> inputs{std::move(audios), std::move(num_rows)};
    auto out =
        encoder_sess_->Run(Ort::RunOptions{nullptr},
                           encoder_input_names_ptr_.data(), inputs.data(),
                           inputs.size(), encoder_output_names_ptr_.data(),
                           encoder_output_names_ptr_.size());
    return std::move(out[0]);
  }

  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLM(Ort::Value input_ids, Ort::Value audio_hidden,
            Ort::Value attention_mask, const Ort::Value &cache_position,
            const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv)
      const {
    const int32_t n = num_layers_;

    std::vector<Ort::Value> inputs;
    inputs.reserve(4 + 2 * n);
    inputs.push_back(std::move(input_ids));
    inputs.push_back(std::move(audio_hidden));
    inputs.push_back(std::move(attention_mask));
    inputs.push_back(Clone(allocator_, &cache_position));
    for (int32_t i = 0; i != n; ++i) {
      inputs.push_back(Clone(allocator_, &cache_kv[i].first));
      inputs.push_back(Clone(allocator_, &cache_kv[i].second));
    }

    auto out =
        decoder_sess_->Run(Ort::RunOptions{nullptr},
                           decoder_input_names_ptr_.data(), inputs.data(),
                           inputs.size(), decoder_output_names_ptr_.data(),
                           decoder_output_names_ptr_.size());

    Ort::Value logits = std::move(out[0]);
    std::vector<std::pair<Ort::Value, Ort::Value>> delta;
    delta.reserve(n);
    for (int32_t i = 0; i != n; ++i) {
      delta.emplace_back(std::move(out[1 + 2 * i]), std::move(out[2 + 2 * i]));
    }
    return {std::move(logits), std::move(delta)};
  }

  std::vector<std::pair<Ort::Value, Ort::Value>> CreateEmptyKVCache(
      int64_t batch) const {
    std::vector<std::pair<Ort::Value, Ort::Value>> cache;
    cache.reserve(num_layers_);
    for (int32_t i = 0; i != num_layers_; ++i) {
      cache.emplace_back(CreateCacheTensor(batch), CreateCacheTensor(batch));
    }
    return cache;
  }

  void ApplyKvDeltaInplace(
      std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
      const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
      const Ort::Value &cache_position) const {
    const auto pos =
        cache_position.GetTensorData<int64_t>()[0];  // first position of step
    const int64_t seq_len =
        kv_delta[0].first.GetTensorTypeAndShapeInfo().GetShape()[2];

    // Cache layout is [1, kv_heads, max_total_len, head_dim], so the slots of
    // one head are contiguous and every head starts at head * max_total_len.
    const size_t head_bytes = static_cast<size_t>(head_dim_) * sizeof(float);
    const size_t head_stride = static_cast<size_t>(max_total_len_) * head_bytes;
    const size_t copy_bytes = static_cast<size_t>(seq_len) * head_bytes;

    for (int32_t i = 0; i != num_layers_; ++i) {
      for (int which = 0; which != 2; ++which) {
        Ort::Value &dst =
            which == 0 ? (*cache_kv)[i].first : (*cache_kv)[i].second;
        const Ort::Value &src =
            which == 0 ? kv_delta[i].first : kv_delta[i].second;

        auto *dst_base =
            static_cast<uint8_t *>(dst.GetTensorMutableData<void>());
        const auto *src_base =
            static_cast<const uint8_t *>(src.GetTensorData<void>());

        for (int32_t h = 0; h != num_kv_heads_; ++h) {
          uint8_t *dst_ptr =
              dst_base + static_cast<size_t>(h) * head_stride +
              static_cast<size_t>(pos) * head_bytes;
          const uint8_t *src_ptr =
              src_base + static_cast<size_t>(h) * copy_bytes;
          std::memcpy(dst_ptr, src_ptr, copy_bytes);
        }
      }
    }
  }

  int32_t GetMaxTotalLen() const { return max_total_len_; }
  int32_t NumLayers() const { return num_layers_; }
  int32_t NumKvHeads() const { return num_kv_heads_; }
  int32_t HeadDim() const { return head_dim_; }
  int32_t VocabSize() const { return vocab_size_; }
  OrtAllocator *Allocator() const { return allocator_; }

 private:
  Ort::Value CreateCacheTensor(int64_t batch) const {
    std::array<int64_t, 4> shape{batch, num_kv_heads_, max_total_len_,
                                 head_dim_};
    auto t = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                             shape.size());
    // Slots that the attention mask hides are still read by the graph, so the
    // cache has to start from zeros instead of whatever the allocator returns.
    const size_t n = static_cast<size_t>(batch) * num_kv_heads_ *
                     max_total_len_ * head_dim_;
    std::fill_n(t.GetTensorMutableData<float>(), n, 0.0f);
    return t;
  }

  void InitEncoder() {
    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);
    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    if (encoder_input_names_.size() != 2) {
      SHERPA_ONNX_LOGE("audio8-asr encoder must have 2 inputs, got %zu",
                       encoder_input_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }
    if (encoder_output_names_.size() != 1) {
      SHERPA_ONNX_LOGE("audio8-asr encoder must have 1 output, got %zu",
                       encoder_output_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void InitDecoder() {
    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);
    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);

    if (decoder_input_names_.size() < 5 ||
        (decoder_input_names_.size() - 4) % 2 != 0) {
      SHERPA_ONNX_LOGE(
          "audio8-asr decoder inputs mismatch: expected 4+2*num_layers, "
          "got %zu",
          decoder_input_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }
    num_layers_ = static_cast<int32_t>((decoder_input_names_.size() - 4) / 2);

    if (decoder_output_names_.size() !=
        static_cast<size_t>(1 + 2 * num_layers_)) {
      SHERPA_ONNX_LOGE(
          "audio8-asr decoder outputs mismatch: expected %d, got %zu",
          1 + 2 * num_layers_, decoder_output_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }

    // cache layout is [1, kv_heads, max_total_len, head_dim]
    auto kv_ti = decoder_sess_->GetInputTypeInfo(4);
    auto kv_shape = kv_ti.GetTensorTypeAndShapeInfo().GetShape();
    if (kv_shape.size() != 4) {
      SHERPA_ONNX_LOGE("audio8-asr decoder cache must be 4-D, got %zu",
                       kv_shape.size());
      SHERPA_ONNX_EXIT(-1);
    }
    num_kv_heads_ = static_cast<int32_t>(kv_shape[1]);
    max_total_len_ = static_cast<int32_t>(kv_shape[2]);
    head_dim_ = static_cast<int32_t>(kv_shape[3]);
    if (num_kv_heads_ <= 0 || max_total_len_ <= 0 || head_dim_ <= 0) {
      SHERPA_ONNX_LOGE(
          "audio8-asr decoder cache shape has dynamic dimensions; please "
          "export with a static max_total_len");
      SHERPA_ONNX_EXIT(-1);
    }

    auto logits_ti = decoder_sess_->GetOutputTypeInfo(0);
    auto logits_shape = logits_ti.GetTensorTypeAndShapeInfo().GetShape();
    vocab_size_ = static_cast<int32_t>(logits_shape.back());
    if (vocab_size_ <= 0) {
      SHERPA_ONNX_LOGE("audio8-asr decoder logits last dim is dynamic");
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_encoder_;
  Ort::SessionOptions sess_opts_decoder_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo cpu_mem_info_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;
  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;
  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  int32_t num_layers_ = 0;
  int32_t num_kv_heads_ = 0;
  int32_t max_total_len_ = 0;
  int32_t head_dim_ = 0;
  int32_t vocab_size_ = 0;
};

OfflineAudio8AsrModel::OfflineAudio8AsrModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineAudio8AsrModel::OfflineAudio8AsrModel(Manager *mgr,
                                             const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineAudio8AsrModel::~OfflineAudio8AsrModel() = default;

Ort::Value OfflineAudio8AsrModel::ForwardEncoder(Ort::Value audios,
                                                 Ort::Value num_rows) const {
  return impl_->ForwardEncoder(std::move(audios), std::move(num_rows));
}

std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
OfflineAudio8AsrModel::ForwardLM(
    Ort::Value input_ids, Ort::Value audio_hidden, Ort::Value attention_mask,
    const Ort::Value &cache_position,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv) const {
  return impl_->ForwardLM(std::move(input_ids), std::move(audio_hidden),
                          std::move(attention_mask), cache_position, cache_kv);
}

std::vector<std::pair<Ort::Value, Ort::Value>>
OfflineAudio8AsrModel::CreateEmptyKVCache(int64_t batch) const {
  return impl_->CreateEmptyKVCache(batch);
}

void OfflineAudio8AsrModel::ApplyKvDeltaInplace(
    std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
    const Ort::Value &cache_position) const {
  impl_->ApplyKvDeltaInplace(cache_kv, kv_delta, cache_position);
}

int32_t OfflineAudio8AsrModel::GetMaxTotalLen() const {
  return impl_->GetMaxTotalLen();
}

int32_t OfflineAudio8AsrModel::NumLayers() const { return impl_->NumLayers(); }

int32_t OfflineAudio8AsrModel::NumKvHeads() const {
  return impl_->NumKvHeads();
}

int32_t OfflineAudio8AsrModel::HeadDim() const { return impl_->HeadDim(); }

int32_t OfflineAudio8AsrModel::VocabSize() const { return impl_->VocabSize(); }

OrtAllocator *OfflineAudio8AsrModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineAudio8AsrModel::OfflineAudio8AsrModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineAudio8AsrModel::OfflineAudio8AsrModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
