// sherpa-onnx/csrc/offline-speech-denoiser-voxlock-impl.h
//
// Copyright (c)  2026  Arbaz Siddiqui

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_IMPL_H_

#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-impl.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/csrc/speech-denoiser-voxlock-stream.h"

namespace sherpa_onnx {

class OfflineSpeechDenoiserVoxLockImpl : public OfflineSpeechDenoiserImpl {
 public:
  explicit OfflineSpeechDenoiserVoxLockImpl(
      const OfflineSpeechDenoiserConfig &config)
      : model_(config.model) {
    Init(config);
  }

  template <typename Manager>
  OfflineSpeechDenoiserVoxLockImpl(Manager *mgr,
                                   const OfflineSpeechDenoiserConfig &config)
      : model_(mgr, config.model) {
    Init(mgr, config);
  }

  DenoisedAudio Run(const float *samples, int32_t n,
                    int32_t sample_rate) const override {
    const auto &meta = model_.GetMetaData();

    std::vector<float> tmp;
    auto p = samples;

    if (sample_rate != meta.sample_rate) {
      SHERPA_ONNX_LOGE(
          "Creating a resampler:\n"
          "   in_sample_rate: %d\n"
          "   output_sample_rate: %d\n",
          sample_rate, meta.sample_rate);

      CreateVoxLockResampler(sample_rate, meta.sample_rate)
          ->Resample(samples, n, true, &tmp);
      p = tmp.data();
      n = tmp.size();
    }

    DenoisedAudio denoised_audio;
    denoised_audio.sample_rate = meta.sample_rate;

    const std::vector<float> *voice_sample = &voice_sample_;
    std::vector<float> selected;
    if (voice_sample->empty()) {
      selected = SelectVoxLockVoiceSample(
          p, n, meta.sample_rate, meta.voice_sample_seconds, kSearchSeconds);
      voice_sample = &selected;
    }

    if (voice_sample->empty()) {
      SHERPA_ONNX_LOGE(
          "The input is shorter than %d seconds, too short to take a voice "
          "sample from: %d samples. Return it unchanged.",
          meta.voice_sample_seconds, n);
      denoised_audio.samples.assign(p, p + n);
      return denoised_audio;
    }

    SpeechDenoiserVoxLockStream stream(&model_, voice_sample->data(),
                                       voice_sample->size());

    denoised_audio.samples = stream.Push(p, n);
    auto tail = stream.Flush();
    denoised_audio.samples.insert(denoised_audio.samples.end(), tail.begin(),
                                  tail.end());
    denoised_audio.samples.resize(n);

    return denoised_audio;
  }

  int32_t GetSampleRate() const override {
    return model_.GetMetaData().sample_rate;
  }

 private:
  void Init(const OfflineSpeechDenoiserConfig &config) {
    const auto &meta = model_.GetMetaData();
    const auto &filename = config.model.voxlock.voice_sample;
    if (!filename.empty()) {
      voice_sample_ = ReadVoxLockVoiceSample(filename, meta.sample_rate,
                                             meta.voice_sample_seconds);
    }
  }

  template <typename Manager>
  void Init(Manager *mgr, const OfflineSpeechDenoiserConfig &config) {
    const auto &meta = model_.GetMetaData();
    const auto &filename = config.model.voxlock.voice_sample;
    if (!filename.empty()) {
      voice_sample_ = ReadVoxLockVoiceSample(mgr, filename, meta.sample_rate,
                                             meta.voice_sample_seconds);
    }
  }

 private:
  // Without a voice sample in the config, the voice sample is the loudest
  // voice_sample_seconds seconds in the first kSearchSeconds seconds
  static constexpr int32_t kSearchSeconds = 15;

  OfflineSpeechDenoiserVoxLockModel model_;
  std::vector<float> voice_sample_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_IMPL_H_
