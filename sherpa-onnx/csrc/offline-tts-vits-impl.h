// sherpa-onnx/csrc/offline-tts-vits-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/lexicon.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsVitsImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsVitsImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(config.model)),
        lexicon_(config.model.vits.lexicon, config.model.vits.tokens,
                 model_->Punctuations(), model_->Language(), config.model.debug,
                 model_->IsPiper()) {
    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
        }
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
      }
    }
  }

#if __ANDROID_API__ >= 9
  OfflineTtsVitsImpl(AAssetManager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(mgr, config.model)),
        lexicon_(mgr, config.model.vits.lexicon, config.model.vits.tokens,
                 model_->Punctuations(), model_->Language(), config.model.debug,
                 model_->IsPiper()) {
    if (!config.rule_fsts.empty()) {
      SHERPA_ONNX_LOGE("TODO(fangjun): Implement rule FST for Android");
    }
  }
#endif

  GeneratedAudio Generate(const std::string &_text, int64_t sid = 0,
                          float speed = 1.0) const override {
    int32_t num_speakers = model_->NumSpeakers();
    if (num_speakers == 0 && sid != 0) {
      SHERPA_ONNX_LOGE(
          "This is a single-speaker model and supports only sid 0. Given sid: "
          "%d. sid is ignored",
          static_cast<int32_t>(sid));
    }

    if (num_speakers != 0 && (sid >= num_speakers || sid < 0)) {
      SHERPA_ONNX_LOGE(
          "This model contains only %d speakers. sid should be in the range "
          "[%d, %d]. Given: %d. Use sid=0",
          num_speakers, 0, num_speakers - 1, static_cast<int32_t>(sid));
      sid = 0;
    }

    std::string text = _text;
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("Raw text: %s", text.c_str());
    }

    if (!tn_list_.empty()) {
      for (const auto &tn : tn_list_) {
        text = tn->Normalize(text);
        if (config_.model.debug) {
          SHERPA_ONNX_LOGE("After normalizing: %s", text.c_str());
        }
      }
    }

    std::vector<int64_t> x = lexicon_.ConvertTextToTokenIds(text);
    if (x.empty()) {
      SHERPA_ONNX_LOGE("Failed to convert %s to token IDs", text.c_str());
      return {};
    }

    if (model_->AddBlank()) {
      std::vector<int64_t> buffer(x.size() * 2 + 1);
      int32_t i = 1;
      for (auto k : x) {
        buffer[i] = k;
        i += 2;
      }
      x = std::move(buffer);
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> x_shape = {1, static_cast<int32_t>(x.size())};
    Ort::Value x_tensor = Ort::Value::CreateTensor(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    Ort::Value audio = model_->Run(std::move(x_tensor), sid, speed);

    std::vector<int64_t> audio_shape =
        audio.GetTensorTypeAndShapeInfo().GetShape();

    int64_t total = 1;
    // The output shape may be (1, 1, total) or (1, total) or (total,)
    for (auto i : audio_shape) {
      total *= i;
    }

    const float *p = audio.GetTensorData<float>();

    GeneratedAudio ans;
    ans.sample_rate = model_->SampleRate();
    ans.samples = std::vector<float>(p, p + total);
    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsVitsModel> model_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
  Lexicon lexicon_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
