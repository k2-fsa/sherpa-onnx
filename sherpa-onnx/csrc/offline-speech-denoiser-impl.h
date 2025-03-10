// sherpa-onnx/csrc/offline-speaker-speech-denoiser-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_IMPL_H_

#include "sherpa-onnx/csrc/offline-speech-denoiser.h"
namespace sherpa_onnx {

class OfflineSpeechDenoiserImpl {
 public:
  virtual ~OfflineSpeechDenoiserImpl() = default;

  static std::unique_ptr<OfflineSpeechDenoiserImpl> Create(
      const OfflineSpeechDenoiserConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineSpeechDenoiserImpl> Create(
      Manager *mgr, const OfflineSpeechDenoiserConfig &config);

  DenoisedAudio Run(const float *samples, int32_t n,
                    int32_t sample_rate) const {
    return {};
  }

  int32_t GetSampleRate() const { return 0; }
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_IMPL_H_
