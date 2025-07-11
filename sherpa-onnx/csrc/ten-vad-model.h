// sherpa-onnx/csrc/ten-vad-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_TEN_VAD_MODEL_H_
#define SHERPA_ONNX_CSRC_TEN_VAD_MODEL_H_

#include <memory>

#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {

class TenVadModel : public VadModel {
 public:
  explicit TenVadModel(const VadModelConfig &config);

  template <typename Manager>
  TenVadModel(Manager *mgr, const VadModelConfig &config);

  ~TenVadModel() override;

  // reset the internal model states
  void Reset() override;

  /**
   * @param samples Pointer to a 1-d array containing audio samples.
   *                Each sample should be normalized to the range [-1, 1].
   * @param n Number of samples.
   *
   * @return Return true if speech is detected. Return false otherwise.
   */
  bool IsSpeech(const float *samples, int32_t n) override;

  // 256 or 160
  int32_t WindowSize() const override;

  // 256 or 128
  int32_t WindowShift() const override;

  int32_t MinSilenceDurationSamples() const override;
  int32_t MinSpeechDurationSamples() const override;

  void SetMinSilenceDuration(float s) override;
  void SetThreshold(float threshold) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TEN_VAD_MODEL_H_
