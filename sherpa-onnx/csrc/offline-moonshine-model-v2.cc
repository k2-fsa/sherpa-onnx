// sherpa-onnx/csrc/offline-moonshine-model-v2.cc
//
// Copyright (c)  2024-2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-moonshine-model-v2.h"

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

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineMoonshineModelV2::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.moonshine.encoder), sess_opts_);
    InitEncoder(nullptr, 0);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.moonshine.merged_decoder),
        sess_opts_);
    InitDecoder(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.moonshine.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.moonshine.merged_decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  Ort::Value ForwardEncoder(Ort::Value audio) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> mask;
    std::vector<Ort::Value> inputs;

    inputs.push_back(std::move(audio));

    if (encoder_input_names_.size() > 1) {
      std::vector<int64_t> shape =
          inputs.back().GetTensorTypeAndShapeInfo().GetShape();

      mask.resize(shape[1], 1);

      Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
          memory_info, mask.data(), mask.size(), shape.data(), shape.size());
      inputs.push_back(std::move(mask_tensor));
    }

    auto features = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());

    return std::move(features[0]);
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> ForwardDecoder(
      Ort::Value tokens, Ort::Value encoder_out,
      std::vector<Ort::Value> states) {
    auto encoder_seq_len = states[2].GetTensorTypeAndShapeInfo().GetShape()[2];
    bool use_cache_branch = encoder_seq_len > 1;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> mask;

    std::vector<Ort::Value> inputs;

    inputs.reserve(4 + states.size());

    if (decoder_needs_mask_) {
      mask.resize(encoder_out.GetTensorTypeAndShapeInfo().GetShape()[1], 1);
      std::array<int64_t, 2> shape = {
          1, encoder_out.GetTensorTypeAndShapeInfo().GetShape()[1]};

      Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
          memory_info, mask.data(), mask.size(), shape.data(), shape.size());

      inputs.push_back(std::move(mask_tensor));
    }

    inputs.push_back(std::move(tokens));
    inputs.push_back(std::move(encoder_out));

    for (auto &s : states) {
      inputs.push_back(View(&s));
    }

    int64_t shape = 1;

    Ort::Value tensor = Ort::Value::CreateTensor<bool>(
        memory_info, &use_cache_branch, 1, &shape, 1);

    inputs.push_back(std::move(tensor));

    auto out = decoder_sess_->Run(
        {}, decoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());

    if (!use_cache_branch) {
      // update encoder and decoder
      for (int32_t i = 0; i < static_cast<int32_t>(states_.size()); ++i) {
        states[i] = std::move(out[1 + i]);
      }
    } else {
      // only update decoder kv
      for (int32_t i = 0; i < num_layers_; ++i) {
        states[4 * i + 0] = std::move(out[1 + 4 * i + 0]);
        states[4 * i + 1] = std::move(out[1 + 4 * i + 1]);
      }
    }

    return {std::move(out[0]), std::move(states)};
  }

  std::vector<Ort::Value> GetDecoderInitStates() {
    std::vector<Ort::Value> ans;

    ans.reserve(states_.size());

    for (auto &s : states_) {
      ans.push_back(View(&s));
    }

    return ans;
  }

  OrtAllocator *Allocator() { return allocator_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!encoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the encoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      decoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!decoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the decoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);

    for (const auto &s : decoder_input_names_) {
      if (Contains(s, "encoder_attention_mask")) {
        decoder_needs_mask_ = true;
      }
    }

    int32_t k = 0;
    for (const auto &s : decoder_input_names_) {
      if (Contains(s, "key_values")) {
        auto shape = decoder_sess_->GetInputTypeInfo(k)
                         .GetTensorTypeAndShapeInfo()
                         .GetShape();
        if (static_cast<int32_t>(shape.size()) != 4) {
          SHERPA_ONNX_LOGE("The shape for %s should be 4-d. Given: %d-d",
                           s.c_str(), static_cast<int32_t>(shape.size()));
          SHERPA_ONNX_EXIT(-1);
        }

        num_head_ = shape[1];
        head_dim_ = shape[3];
        break;
      }
      k += 1;
    }

    if (decoder_needs_mask_) {
      // [ mask, ids, encoder_out, states, use_cache_branch]
      num_layers_ = (static_cast<int32_t>(decoder_input_names_.size()) - 4) / 4;
    } else {
      // [ ids, encoder_out, states, use_cache_branch]
      num_layers_ = (static_cast<int32_t>(decoder_input_names_.size()) - 3) / 4;
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("need attention mask: %d",
                       static_cast<int32_t>(decoder_needs_mask_));
      SHERPA_ONNX_LOGE("num_head: %d", num_head_);
      SHERPA_ONNX_LOGE("head_dim: %d", head_dim_);
      SHERPA_ONNX_LOGE("num_layers: %d", num_layers_);
    }

    InitDecoderStates();
  }

  void InitDecoderStates() {
    states_.reserve(num_layers_ * 4);
    std::array<int64_t, 4> shape{1, num_head_, 0, head_dim_};

    auto n = shape[0] * shape[1] * shape[2] * shape[3];

    for (int32_t i = 0; i < 4 * num_layers_; ++i) {
      Ort::Value v = Ort::Value::CreateTensor<float>(Allocator(), shape.data(),
                                                     shape.size());

      float *p = v.GetTensorMutableData<float>();
      memset(p, 0, sizeof(float) * n);
      states_.push_back(std::move(v));
    }
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

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

  std::vector<Ort::Value> states_;

  int32_t num_head_ = 0;
  int32_t head_dim_ = 0;
  int32_t num_layers_ = 0;
  bool decoder_needs_mask_ = false;
};

OfflineMoonshineModelV2::OfflineMoonshineModelV2(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineMoonshineModelV2::OfflineMoonshineModelV2(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineMoonshineModelV2::~OfflineMoonshineModelV2() = default;

Ort::Value OfflineMoonshineModelV2::ForwardEncoder(Ort::Value audio) const {
  return impl_->ForwardEncoder(std::move(audio));
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OfflineMoonshineModelV2::ForwardDecoder(Ort::Value token,
                                        Ort::Value encoder_out,
                                        std::vector<Ort::Value> states) const {
  return impl_->ForwardDecoder(std::move(token), std::move(encoder_out),
                               std::move(states));
}

std::vector<Ort::Value> OfflineMoonshineModelV2::GetDecoderInitStates() const {
  return impl_->GetDecoderInitStates();
}

OrtAllocator *OfflineMoonshineModelV2::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineMoonshineModelV2::OfflineMoonshineModelV2(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineMoonshineModelV2::OfflineMoonshineModelV2(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
