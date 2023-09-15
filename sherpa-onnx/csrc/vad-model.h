// sherpa-onnx/csrc/vad-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VAD_MODEL_H_
#define SHERPA_ONNX_CSRC_VAD_MODEL_H_

#include <memory>

#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

class VadModel {
 public:
  virtual ~VadModel() = default;

  static std::unique_ptr<VadModel> Create(const VadModelConfig &config);

  // reset the internal model states
  virtual void Reset() = 0;

  /**
   * @param samples Pointer to a 1-d array containing audio samples.
   *                Each sample should be normalized to the range [-1, 1].
   * @param n Number of samples.
   *
   * @return Return true if speech is detected. Return false otherwise.
   */
  virtual bool IsSpeech(const float *samples, int32_t n) = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VAD_MODEL_H_
