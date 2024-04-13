// sherpa-onnx/csrc/offline-ct-transformer-model.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_CT_TRANSFORMER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CT_TRANSFORMER_MODEL_H_
#include <memory>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-ct-transformer-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-punctuation-model-config.h"

namespace sherpa_onnx {

/** This class implements
 * https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/python/onnxruntime/funasr_onnx/punc_bin.py#L17
 * from FunASR
 */
class OfflineCtTransformerModel {
 public:
  explicit OfflineCtTransformerModel(
      const OfflinePunctuationModelConfig &config);

#if __ANDROID_API__ >= 9
  OfflineCtTransformerModel(AAssetManager *mgr,
                            const OfflinePunctuationModelConfig &config);
#endif

  ~OfflineCtTransformerModel();

  /** Run the forward method of the model.
   *
   * @param text  A tensor of shape (N, T) of dtype int32.
   * @param text  A tensor of shape (N) of dtype int32.
   *
   * @return Return a tensor
   *  - punctuation_ids: A 2-D tensor of shape (N, T).
   */
  Ort::Value Forward(Ort::Value text, Ort::Value text_len) const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

  const OfflineCtTransformerModelMetaData &GetModelMetadata() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CT_TRANSFORMER_MODEL_H_
