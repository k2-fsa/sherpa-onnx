// sherpa-onnx/csrc/offline-tts-piper-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-piper-model.h"
#include "sherpa-onnx/csrc/piper-voice.h"


namespace sherpa_onnx {

class OfflineTtsPiperImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsPiperImpl(const OfflineTtsConfig &config);

#if __ANDROID_API__ >= 9
  OfflineTtsPiperImpl(AAssetManager *mgr, const OfflineTtsConfig &config);
#endif

  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override;

  int32_t SampleRate() const override;

  int32_t NumSpeakers() const override;

 private:
  void InitFrontend();
  
  bool LoadVoiceConfig();
  
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsPiperModel> model_;
  std::unique_ptr<OfflineTtsFrontend> frontend_;
  
  mutable sherpa_onnx::piper::Voice voice_data_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_IMPL_H_