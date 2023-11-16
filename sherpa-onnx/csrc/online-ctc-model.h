// sherpa-onnx/csrc/online-ctc-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_MODEL_H_

#include <memory>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"

namespace sherpa_onnx {

class OnlineCtcModel {
 public:
  virtual ~OnlineCtcModel() = default;

  static std::unique_ptr<OnlineCtcModel> Create(
      const OnlineModelConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<OnlineCtcModel> Create(
      AAssetManager *mgr, const OnlineModelConfig &config);
#endif

  // Return a list of tensors containing the initial states
  virtual std::vector<Ort::Value> GetInitStates() const = 0;

  /**
   *
   * @param x A 3-D tensor of shape (N, T, C). N has to be 1.
   * @param states  It is from GetInitStates() or returned from this method.
   *
   * @return Return a list of tensors
   *    - ans[0] contains log_probs, of shape (N, T, C)
   *    - ans[1:] contains next_states
   */
  virtual std::vector<Ort::Value> Forward(
      Ort::Value x, std::vector<Ort::Value> states) const = 0;

  /** Return the vocabulary size of the model
   */
  virtual int32_t VocabSize() const = 0;

  /** Return an allocator for allocating memory
   */
  virtual OrtAllocator *Allocator() const = 0;

  // The model accepts this number of frames before subsampling as input
  virtual int32_t ChunkLength() const = 0;

  // Similar to frame_shift in feature extractor, after processing
  // ChunkLength() frames, we advance by ChunkShift() frames
  // before we process the next chunk.
  virtual int32_t ChunkShift() const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_MODEL_H_
