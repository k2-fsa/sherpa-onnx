// sherpa-onnx/csrc/vad-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VAD_MODEL_H_
#define SHERPA_ONNX_CSRC_VAD_MODEL_H_

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

class VadModel {
 public:
  virtual ~VadModel() = default;

  static std::unique_ptr<VadModel> Create(const VadModelConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<VadModel> Create(AAssetManager *mgr,
                                          const VadModelConfig &config);
#endif

  // reset the internal model states
  virtual void Reset() = 0;

  /**
   * @param samples Pointer to a 1-d array containing audio samples.
   *                Each sample should be normalized to the range [-1, 1].
   * @param n Number of samples. Should be equal to WindowSize()
   *
   * @return Return true if speech is detected. Return false otherwise.
   */
  virtual bool IsSpeech(const float *samples, int32_t n) = 0;

  virtual int32_t WindowSize() const = 0;

  virtual int32_t MinSilenceDurationSamples() const = 0;
  virtual int32_t MinSpeechDurationSamples() const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VAD_MODEL_H_
