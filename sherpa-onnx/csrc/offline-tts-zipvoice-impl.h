// sherpa-onnx/csrc/offline-tts-zipvoice-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_

#include <cmath>
#include <memory>
#include <string>
#include <strstream>
#include <utility>
#include <vector>

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/jieba-lexicon.h"
#include "sherpa-onnx/csrc/lexicon.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/melo-tts-lexicon.h"
#include "sherpa-onnx/csrc/offline-tts-character-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/vocoder.h"

namespace sherpa_onnx {

class OfflineTtsZipvoiceImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsZipvoiceImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsZipvoiceModel>(config.model)),
        vocoder_(Vocoder::Create(config.model)) {
    InitFrontend();

  template <typename Manager>
  OfflineTtsZipvoiceImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsZipvoiceModel>(mgr, config.model)),
        vocoder_(Vocoder::Create(mgr, config.model)) {
    InitFrontend(mgr);
  } 

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  GeneratedAudio Generate(const std::string& text,
                          const std::string& prompt_text,
                          const std::vector<float>& prompt_samples,
                          float speed, int num_step) const {

    std::vector<TokenIDs> text_token_ids =
        frontend_->ConvertTextToTokenIds(text);

    std::vector<TokenIDs> prompt_token_ids =
        frontend_->ConvertTextToTokenIds(prompt_text);

    if (text_token_ids.empty() ||
        (text_token_ids.size() == 1 && text_token_ids[0].tokens.empty())) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Failed to convert '%{public}s' to token IDs", text.c_str());
#else
      SHERPA_ONNX_LOGE("Failed to convert '%s' to token IDs", text.c_str());
#endif
      return {};
    }

    if (prompt_token_ids.empty() ||
        (prompt_token_ids.size() == 1 && prompt_token_ids[0].tokens.empty())) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Failed to convert prompt text '%{public}s' to token IDs",
                       prompt_text.c_str());
#else
      SHERPA_ONNX_LOGE("Failed to convert prompt text '%s' to token IDs",
                       prompt_text.c_str());
#endif
      return {};
    }

    // we assume batch size is 1
    std::vector<int64_t> tokens = text_token_ids[0].tokens;
    std::vector<int64_t> prompt_tokens = prompt_token_ids[0].tokens;

    return Process(tokens, prompt_tokens, prompt_samples, speed, num_step);
  }


 private:
  template <typename Manager>
  void InitFrontend(Manager *mgr) {
    // TODO:
  }

  void InitFrontend() {
    // TODO: 
  }

  GeneratedAudio Process(const std::vector<int64_t> &tokens,
                       const std::vector<int64_t> &prompt_tokens,
                       const std::vector<float> &prompt_samples,
                       float speed, int num_step) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> tokens_shape = {1, static_cast<int64_t>(tokens.size())};
    Ort::Value tokens_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<int64_t*>(tokens.data()),
        tokens.size(), tokens_shape.data(), tokens_shape.size());

    std::array<int64_t, 2> prompt_tokens_shape =
        {1, static_cast<int64_t>(prompt_tokens.size())};
    Ort::Value prompt_tokens_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<int64_t*>(prompt_tokens.data()),
        prompt_tokens.size(), prompt_tokens_shape.data(), prompt_tokens_shape.size());

    float target_rms = config_.model.zipvoice.target_rms;
    float feat_scale = config_.model.zipvoice.feat_scale;

    // Scale prompt_samples
    float prompt_rms = 0.0f;
    // Compute RMS of prompt_samples
    double sum_sq = 0.0;
    for (float s : prompt_samples) {
        sum_sq += s * s;
    }
    prompt_rms = std::sqrt(sum_sq / prompt_samples.size());
    if (prompt_rms < target_rms && prompt_rms > 0.0f) {
        float scale = target_rms / static_cast<float>(prompt_rms);
        for (auto& s : prompt_samples) {
            s *= scale;
        }
    }

    // TODO: implement ExtractMelSpectrogram, return shape (mel_dim, num_frames)
    std::vector<std::vector<float>> prompt_features =
        ExtractMelSpectrogram(prompt_samples);
    const int mel_dim = prompt_features.size();
    const int num_frames = mel_dim > 0 ? prompt_features[0].size() : 0;

    if (feat_scale != 1.0f) {
        for (auto& row : prompt_features) {
            for (auto& v : row) {
                v *= feat_scale;
            }
        }
    }

    // Convert the 2D feature matrix into a contiguous 1D array for tensor input
    std::vector<float> prompt_features_flat;
    prompt_features_flat.reserve(mel_dim * num_frames);
    for (int i = 0; i < mel_dim; ++i) {
        for (int j = 0; j < num_frames; ++j) {
            prompt_features_flat.push_back(prompt_features[i][j]);
        }
    }

    std::array<int64_t, 3> shape = {1, mel_dim, num_frames};
    prompt_features_tensor = Ort::Value::CreateTensor(
        memory_info, prompt_features_flat.data(), prompt_features_flat.size(),
        shape.data(), shape.size());

    Ort::Value mel = model_->Run(
        tokens_tensor, prompt_tokens_tensor, prompt_features_tensor,
        speed, num_step);
    
    // permute tensor and scale it
    std::vector<int64_t> mel_shape = mel.GetTensorTypeAndShapeInfo().GetShape();
    int64_t B = mel_shape[0], T = mel_shape[1], C = mel_shape[2];

    float* mel_data = mel.GetTensorMutableData<float>();
    std::vector<float> mel_permuted(B * C * T);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t t = 0; t < T; ++t) {
                int64_t src_idx = b * T * C + t * C + c;
                int64_t dst_idx = b * C * T + c * T + t;
                mel_permuted[dst_idx] = mel_data[src_idx] / feat_scale;
            }
        }
    }

    std::array<int64_t, 3> new_shape = {B, C, T};
    Ort::Value mel_new = Ort::Value::CreateTensor<float>(
        memory_info, mel_permuted.data(), mel_permuted.size(),
        new_shape.data(), new_shape.size());

    GeneratedAudio ans;
    ans.samples = vocoder_->Run(std::move(mel_new));
    ans.sample_rate = model_->GetMetaData().sample_rate;

    if (prompt_rms < target_rms && target_rms > 0.0f) {
        float scale = prompt_rms / target_rms;
        for (auto& s : ans.samples) {
            s *= scale;
        }
    }
    
    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsZipvoiceModel> model_;
  std::unique_ptr<Vocoder> vocoder_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_IMPL_H_
