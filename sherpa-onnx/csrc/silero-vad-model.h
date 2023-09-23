// sherpa-onnx/csrc/silero-vad-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_H_
#define SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_H_

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {

class SileroVadModel : public VadModel {
 public:
  explicit SileroVadModel(const VadModelConfig &config);

#if __ANDROID_API__ >= 9
  SileroVadModel(AAssetManager *mgr, const VadModelConfig &config);
#endif

  ~SileroVadModel() override;

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

  int32_t WindowSize() const override;

  int32_t MinSilenceDurationSamples() const override;
  int32_t MinSpeechDurationSamples() const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SILERO_VAD_MODEL_H_
