// sherpa-onnx/csrc/online-zipformer2-ctc-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER2_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER2_CTC_MODEL_H_

#include <memory>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-ctc-model.h"
#include "sherpa-onnx/csrc/online-model-config.h"

namespace sherpa_onnx {

class OnlineZipformer2CtcModel : public OnlineCtcModel {
 public:
  explicit OnlineZipformer2CtcModel(const OnlineModelConfig &config);

#if __ANDROID_API__ >= 9
  OnlineZipformer2CtcModel(AAssetManager *mgr, const OnlineModelConfig &config);
#endif

  ~OnlineZipformer2CtcModel() override;

  // A list of tensors.
  // See also
  // https://github.com/k2-fsa/icefall/pull/1413
  // and
  // https://github.com/k2-fsa/icefall/pull/1415
  std::vector<Ort::Value> GetInitStates() const override;

  std::vector<Ort::Value> StackStates(
      std::vector<std::vector<Ort::Value>> states) const override;

  std::vector<std::vector<Ort::Value>> UnStackStates(
      std::vector<Ort::Value> states) const override;

  /**
   *
   * @param x A 3-D tensor of shape (N, T, C). N has to be 1.
   * @param states  It is from GetInitStates() or returned from this method.
   *
   * @return Return a list of tensors
   *    - ans[0] contains log_probs, of shape (N, T, C)
   *    - ans[1:] contains next_states
   */
  std::vector<Ort::Value> Forward(
      Ort::Value x, std::vector<Ort::Value> states) const override;

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const override;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const override;

  // The model accepts this number of frames before subsampling as input
  int32_t ChunkLength() const override;

  // Similar to frame_shift in feature extractor, after processing
  // ChunkLength() frames, we advance by ChunkShift() frames
  // before we process the next chunk.
  int32_t ChunkShift() const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER2_CTC_MODEL_H_
