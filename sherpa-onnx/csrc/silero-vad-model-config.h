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
  float prob = 0.5;

  float min_silence_duration = 0.1;  // in seconds

  // 512, 1024, 1536 samples for 16000 Hz
  // 256, 512, 768 samples for 800 Hz
  int window_size = 1536;  // in samples

  // support only 16000 and 8000
  int32_t sample_rate = 16000;

  SileroVadModelConfig() = default;

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_CONFIG_H_
