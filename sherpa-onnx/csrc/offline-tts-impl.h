// sherpa-onnx/csrc/offline-tts-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/offline-tts.h"

namespace sherpa_onnx {

class OfflineTtsImpl {
 public:
  virtual ~OfflineTtsImpl() = default;

  static std::unique_ptr<OfflineTtsImpl> Create(const OfflineTtsConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineTtsImpl> Create(Manager *mgr,
                                                const OfflineTtsConfig &config);

  virtual GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const = 0;

  // Return the sample rate of the generated audio
  virtual int32_t SampleRate() const = 0;

  // Number of supported speakers.
  // If it supports only a single speaker, then it return 0 or 1.
  virtual int32_t NumSpeakers() const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_
