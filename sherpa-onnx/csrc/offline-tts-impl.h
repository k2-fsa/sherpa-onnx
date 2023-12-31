// sherpa-onnx/csrc/offline-tts-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_

#include <memory>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/offline-tts.h"

namespace sherpa_onnx {

class OfflineTtsImpl {
 public:
  virtual ~OfflineTtsImpl() = default;

  static std::unique_ptr<OfflineTtsImpl> Create(const OfflineTtsConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<OfflineTtsImpl> Create(AAssetManager *mgr,
                                                const OfflineTtsConfig &config);
#endif

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
