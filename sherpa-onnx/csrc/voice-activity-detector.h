// sherpa-onnx/csrc/voice-activity-detector.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
#define SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

struct SpeechSegment {
  // Start of this segment in the input stream, in samples.
  //
  // It is an absolute position and keeps increasing during the lifetime of
  // the detector, so it is 64-bit. 2^63 samples is about 18 million years
  // at 16 kHz.
  //
  // It is -1 if this segment is not valid.
  int64_t start = -1;
  std::vector<float> samples;
};

class VoiceActivityDetector {
 public:
  explicit VoiceActivityDetector(const VadModelConfig &config,
                                 float buffer_size_in_seconds = 60);

  template <typename Manager>
  VoiceActivityDetector(Manager *mgr, const VadModelConfig &config,
                        float buffer_size_in_seconds = 60);

  ~VoiceActivityDetector();

  void AcceptWaveform(const float *samples, int32_t n);
  float Compute(const float *samples, int32_t n);

  bool Empty() const;
  void Pop();
  void Clear();

  // It is an error to call Front() if Empty() returns true.
  //
  // The returned reference is valid until the next call to any
  // methods of VoiceActivityDetector.
  const SpeechSegment &Front() const;

  bool IsSpeechDetected() const;

  // It is empty if IsSpeechDetected() returns false
  SpeechSegment CurrentSpeechSegment() const;

  void Reset() const;

  // At the end of the utterance, you can invoke this method so that
  // the last speech segment can be detected.
  void Flush() const;

  const VadModelConfig &GetConfig() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
