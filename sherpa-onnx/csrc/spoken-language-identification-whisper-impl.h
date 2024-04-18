// sherpa-onnx/csrc/spoken-language-identification-whisper-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_WHISPER_IMPL_H_
#define SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_WHISPER_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/offline-whisper-model.h"
#include "sherpa-onnx/csrc/spoken-language-identification-impl.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class SpokenLanguageIdentificationWhisperImpl
    : public SpokenLanguageIdentificationImpl {
 public:
  explicit SpokenLanguageIdentificationWhisperImpl(
      const SpokenLanguageIdentificationConfig &config)
      : config_(config), model_(std::make_unique<OfflineWhisperModel>(config)) {
    Check();
  }

#if __ANDROID_API__ >= 9
  SpokenLanguageIdentificationWhisperImpl(
      AAssetManager *mgr, const SpokenLanguageIdentificationConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineWhisperModel>(mgr, config)) {
    Check();
  }
#endif

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(WhisperTag{});
  }

  std::string Compute(OfflineStream *s) const override {
    int32_t max_num_frames = 3000;
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();
    int32_t num_frames = f.size() / feat_dim;

    // we use 50 here so that there will be some zero tail paddings
    if (num_frames >= max_num_frames - 50) {
      SHERPA_ONNX_LOGE(
          "Only waves less than 30 seconds are supported. We process only the "
          "first 30 seconds and discard the remaining data");
      num_frames = max_num_frames - 50;
    }

    model_->NormalizeFeatures(f.data(), num_frames, feat_dim);

    // note that 1000 is an experience-value.
    // You can replace 1000 by other values, say, 100.
    //
    // Since we have removed the 30 seconds constraint, we need
    // tail_padding_frames so that whisper is able to detect the eot token.
    int32_t tail_padding_frames = 1000;

    if (config_.whisper.tail_paddings > 0) {
      tail_padding_frames = config_.whisper.tail_paddings;
    }

    int32_t actual_frames =
        std::min(num_frames + tail_padding_frames, max_num_frames);

    std::array<int64_t, 3> shape{1, actual_frames, feat_dim};

    Ort::Value mel = Ort::Value::CreateTensor<float>(
        model_->Allocator(), shape.data(), shape.size());

    float *p_mel = mel.GetTensorMutableData<float>();
    std::copy(f.data(), f.data() + num_frames * feat_dim, p_mel);

    std::fill_n(p_mel + num_frames * feat_dim,
                (actual_frames - num_frames) * feat_dim, 0);

    mel = Transpose12(model_->Allocator(), &mel);

    try {
      auto cross_kv = model_->ForwardEncoder(std::move(mel));
      int32_t lang_id = model_->DetectLanguage(cross_kv.first, cross_kv.second);
      const auto &id2lang = model_->GetID2Lang();
      if (id2lang.count(lang_id)) {
        return id2lang.at(lang_id);
      } else {
        SHERPA_ONNX_LOGE("Unknown language ID: %d. Return an empty string.",
                         lang_id);
        return "";
      }
    } catch (const Ort::Exception &ex) {
      SHERPA_ONNX_LOGE(
          "\n\nCaught exception:\n\n%s\n\nReturn an empty result. Number of "
          "input frames: %d, Current tail "
          "paddings: %d. If you see a lot of such exceptions, please consider "
          "using a larger --whisper-tail-paddings",
          ex.what(), num_frames, tail_padding_frames);
      return "";
    }
  }

 private:
  void Check() const {
    if (!model_->IsMultiLingual()) {
      SHERPA_ONNX_LOGE(
          "Only whisper multilingual models can be used for spoken language "
          "identification. Given: %s,%s",
          config_.whisper.encoder.c_str(), config_.whisper.decoder.c_str());
      exit(-1);
    }
  }

 private:
  SpokenLanguageIdentificationConfig config_;
  std::unique_ptr<OfflineWhisperModel> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_WHISPER_IMPL_H_
