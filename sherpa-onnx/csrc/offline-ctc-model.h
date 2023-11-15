// sherpa-onnx/csrc/offline-ctc-model.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CTC_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineCtcModel {
 public:
  virtual ~OfflineCtcModel() = default;

  static std::unique_ptr<OfflineCtcModel> Create(
      const OfflineModelConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<OfflineCtcModel> Create(
      AAssetManager *mgr, const OfflineModelConfig &config);
#endif

  /** Run the forward method of the model.
   *
   * @param features  A tensor of shape (N, T, C).
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int64_t.
   *
   * @return Return a vector containing:
   *  - log_probs: A 3-D tensor of shape (N, T', vocab_size).
   *  - log_probs_length A 1-D tensor of shape (N,). Its dtype is int64_t
   */
  virtual std::vector<Ort::Value> Forward(Ort::Value features,
                                          Ort::Value features_length) = 0;

  /** Return the vocabulary size of the model
   */
  virtual int32_t VocabSize() const = 0;

  /** SubsamplingFactor of the model
   *
   * For NeMo Citrinet, the subsampling factor is usually 4.
   * For NeMo Conformer CTC, the subsampling factor is usually 8.
   */
  virtual int32_t SubsamplingFactor() const { return 1; }

  /** Return an allocator for allocating memory
   */
  virtual OrtAllocator *Allocator() const = 0;

  /** For some models, e.g., those from NeMo, they require some preprocessing
   * for the features.
   */
  virtual std::string FeatureNormalizationMethod() const { return {}; }

  // Return true if the model supports batch size > 1
  virtual bool SupportBatchProcessing() const { return true; }
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CTC_MODEL_H_
