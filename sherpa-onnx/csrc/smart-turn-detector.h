// sherpa-onnx/csrc/smart-turn-detector.h
//
// Copyright (c) 2026 Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SMART_TURN_DETECTOR_H_
#define SHERPA_ONNX_CSRC_SMART_TURN_DETECTOR_H_

#include <memory>

#include "sherpa-onnx/csrc/smart-turn-config.h"
#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

class SmartTurnDetector {
 public:
  SmartTurnDetector(const SmartTurnConfig &config, int32_t sample_rate,
                    int32_t num_threads, const std::string &provider,
                    bool debug);
  ~SmartTurnDetector();

  float Compute(const float *samples, int32_t n) const;
  bool IsEndOfTurn(const float *samples, int32_t n) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SMART_TURN_DETECTOR_H_