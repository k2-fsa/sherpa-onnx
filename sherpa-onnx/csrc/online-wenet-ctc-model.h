// sherpa-onnx/csrc/online-wenet-ctc-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_WENET_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_WENET_CTC_MODEL_H_

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

class OnlineWenetCtcModel {
 public:
  explicit OnlineWenetCtcModel(const OnlineModelConfig &config);

#if __ANDROID_API__ >= 9
  OnlineWenetCtcModel(AAssetManager *mgr, const OnlineModelConfig &config);
#endif

  ~OnlineWenetCtcModel();

  std::vector<Ort::Value> GetInitStates();

  /**
   *
   * @param x A 3-D tensor of shape (N, T, C). N has to be 1.
   * @param offset A scalar tensor of type int64.
   * @param states  It is from GetInitStates() or returned from this method.
   *
   * @return Return a list of tensors
   *    - ans[0] contains log_probs, of shape (N, T, C)
   *    - ans[1:] contains next_states
   */
  std::vector<Ort::Value> Forward(Ort::Value x, Ort::Value offset,
                                  std::vector<Ort::Value> states) const;

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

  // The model accepts this number of frames before subsampling as input
  int32_t ChunkLength() const;

  // Similar to frame_shift in feature extractor, after processing
  // ChunkLength() frames, we advance by ChunkShift() frames
  // before we process the next chunk.
  int32_t ChunkShift() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_WENET_CTC_MODEL_H_
