// sherpa-onnx/csrc/silero-vad-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct SileroVadModelConfig {
  std::string model;

  // threshold to classify a segment as speech
  //
  // The predicted probability of a segment is larger than this
  // value, then it is classified as speech.
  float threshold = 0.5;

  float min_silence_duration = 0.5;  // in seconds

  float min_speech_duration = 0.25;  // in seconds

  // 512, 1024, 1536 samples for 16000 Hz
  // 256, 512, 768 samples for 800 Hz
  int window_size = 512;  // in samples

  SileroVadModelConfig() = default;

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_CONFIG_H_
