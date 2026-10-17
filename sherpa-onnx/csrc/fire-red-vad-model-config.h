// sherpa-onnx/csrc/fire-red-vad-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_FIRE_RED_VAD_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_FIRE_RED_VAD_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct FireRedVadModelConfig {
  std::string model;

  // threshold to classify a segment as speech
  //
  // If the predicted probability of a segment is larger than this
  // value, then it is classified as speech.
  float threshold = 0.5;

  float min_silence_duration = 0.5;  // in seconds

  float min_speech_duration = 0.25;  // in seconds

  int32_t window_size = 400;  // in samples

  // If a speech segment is longer than this value, then we increase
  // the threshold to 0.9. After finishing detecting the segment,
  // the threshold value is reset to its original value.
  float max_speech_duration = 20;  // in seconds

  // Negative (exit) threshold for transitioning from speech → silence.
  // If left as a negative value, the default FireRedVAD rule applies:
  //     neg_threshold = max(threshold - 0.15f, 0.01f)
  // This prevents the exit threshold from becoming negative when
  // threshold < 0.15.
  float neg_threshold = -1;

  FireRedVadModelConfig() = default;

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FIRE_RED_VAD_MODEL_CONFIG_H_
