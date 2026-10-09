// sherpa-onnx/csrc/smart-turn-config.h
//
// Copyright (c) 2026 Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SMART_TURN_CONFIG_H_
#define SHERPA_ONNX_CSRC_SMART_TURN_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct SmartTurnConfig {
  // ONNX model taking a float32 mono waveform shaped [1, num_samples] and
  // returning an end-of-turn probability as its first float output.
  // wget https://huggingface.co/soniqo/Smart-Turn-v3.2-ONNX/resolve/main/smart-turn-v3.2-int8.onnx
  std::string model;

  float threshold = 0.5;

  // Sample rate expected by the Smart Turn model.
  int32_t sample_rate = 16000;

  // Audio duration supplied to the model. Audio shorter than this is padded
  // with zeros and longer audio keeps its most recent samples.
  float window_size = 8.0;

  // Run Smart Turn after this much VAD-confirmed trailing silence. A segment
  // is always finalized after max_silence_duration.
  float min_silence_duration = 0.1;
  float max_silence_duration = 1.5;

  SmartTurnConfig() = default;

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SMART_TURN_CONFIG_H_