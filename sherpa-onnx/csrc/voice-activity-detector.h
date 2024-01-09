// sherpa-onnx/csrc/voice-activity-detector.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
#define SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_

#include <memory>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

struct SpeechSegment {
  int32_t start;  // in samples
  std::vector<float> samples;
};

class VoiceActivityDetector {
 public:
  explicit VoiceActivityDetector(const VadModelConfig &config,
                                 float buffer_size_in_seconds = 60);

#if __ANDROID_API__ >= 9
  VoiceActivityDetector(AAssetManager *mgr, const VadModelConfig &config,
                        float buffer_size_in_seconds = 60);
#endif

  ~VoiceActivityDetector();

  void AcceptWaveform(const float *samples, int32_t n);
  bool Empty() const;
  void Pop();
  void Clear();
  const SpeechSegment &Front() const;

  bool IsSpeechDetected() const;

  void Reset();

  const VadModelConfig &GetConfig() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
