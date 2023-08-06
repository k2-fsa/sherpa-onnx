// sherpa-onnx/csrc/offline-recognizer-whisper-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-whisper-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineRecognizerWhisperImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerWhisperImpl(const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineWhisperModel>(config.model_config)) {
    symbol_table_.ApplyBase64Decode();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(WhisperTag{});
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    // batch decoding is not implemented yet
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

 private:
  void DecodeStream(OfflineStream *s) const {
    int32_t max_num_frames = 3000;
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();
    int32_t num_frames = f.size() / feat_dim;

    if (num_frames > 3000) {
      SHERPA_ONNX_LOG(FATAL)
          << "Only waves less than 30 seconds are supported.";
    }

    NormalizeFeatures(f.data(), num_frames, feat_dim);

    std::array<int64_t, 3> shape{1, max_num_frames, feat_dim};

    Ort::Value mel = Ort::Value::CreateTensor<float>(
        model_->Allocator(), shape.data(), shape.size());
    float *p_mel = mel.GetTensorMutableData<float>();
    std::copy(f.begin(), f.end(), p_mel);

    memset(p_mel + f.size(), 0,
           (max_num_frames - num_frames) * feat_dim * sizeof(float));
    mel = Transpose12(model_->Allocator(), &mel);

    auto cross_kv = model_->ForwardEncoder(std::move(mel));
    auto self_kv_cache = model_->GetInitialSelfKVCache();

    const std::vector<int32_t> &initial_tokens = model_->GetInitialTokens();

    std::array<int64_t, 2> token_shape{
        1, static_cast<int64_t>(initial_tokens.size())};

    Ort::Value tokens = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), token_shape.data(), token_shape.size());

    int64_t *p_tokens = tokens.GetTensorMutableData<int64_t>();

    std::copy(initial_tokens.begin(), initial_tokens.end(), p_tokens);

    std::array<int64_t, 1> offset_shape{1};
    Ort::Value offset = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), offset_shape.data(), offset_shape.size());
    *(offset.GetTensorMutableData<int64_t>()) = 0;

    auto decoder_out = model_->ForwardDecoder(
        std::move(tokens), std::move(self_kv_cache.first),
        std::move(self_kv_cache.second), std::move(cross_kv.first),
        std::move(cross_kv.second), std::move(offset));

    auto &logits = std::get<0>(decoder_out);
    const float *p_logits = logits.GetTensorData<float>();

    auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    int32_t vocab_size = logits_shape[2];

    int32_t max_token_id = static_cast<int32_t>(std::distance(
        p_logits, std::max_element(p_logits, p_logits + vocab_size)));

    std::vector<int32_t> results;
    for (int32_t i = 0; i < 400; ++i) {
      if (max_token_id == model_->EOT()) {
        break;
      }
      results.push_back(max_token_id);

      std::array<int64_t, 2> token_shape{1, 1};
      Ort::Value tokens = Ort::Value::CreateTensor<int64_t>(
          model_->Allocator(), token_shape.data(), token_shape.size());
      int64_t *p_tokens = tokens.GetTensorMutableData<int64_t>();
      p_tokens[0] = max_token_id;

      int64_t *p_offset =
          std::get<5>(decoder_out).GetTensorMutableData<int64_t>();

      if (i == 0) {
        *p_offset = initial_tokens.size();
      } else {
        *p_offset += 1;
      }

      decoder_out = model_->ForwardDecoder(std::move(tokens),
                                           std::move(std::get<1>(decoder_out)),
                                           std::move(std::get<2>(decoder_out)),
                                           std::move(std::get<3>(decoder_out)),
                                           std::move(std::get<4>(decoder_out)),
                                           std::move(std::get<5>(decoder_out)));

      auto &logits = std::get<0>(decoder_out);
      const float *p_logits = logits.GetTensorData<float>();

      max_token_id = static_cast<int64_t>(std::distance(
          p_logits, std::max_element(p_logits, p_logits + vocab_size)));
    }

    std::string sr;
    for (auto i : results) {
      if (symbol_table_.contains(i)) {
        sr += symbol_table_[i];
      }
    }
    SHERPA_ONNX_LOGE("%s\n", sr.c_str());
  }

 private:
  static void NormalizeFeatures(float *features, int32_t num_frames,
                                int32_t feat_dim) {
    // log_spec = torch.clamp(features, min=1e-10).log10()
    // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    // mel = (log_spec + 4.0) / 4.0

    int32_t n = num_frames * feat_dim;
    float max_v = -1e20;
    for (int32_t i = 0; i != n; ++i) {
      float f = features[i];

      f = std::max<float>(f, 1e-10);
      f = std::log10(f);

      max_v = std::max(f, max_v);

      features[i] = f;
    }

    max_v -= 8;

    for (int32_t i = 0; i != n; ++i) {
      float f = features[i];
      f = std::max(f, max_v);

      f = (f + 4) / 4;

      features[i] = f;
    }
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineWhisperModel> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
