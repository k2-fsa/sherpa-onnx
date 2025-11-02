// sherpa-onnx/csrc/ascend/offline-paraformer-model-ascend.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ASCEND_OFFLINE_PARAFORMER_MODEL_ASCEND_H_
#define SHERPA_ONNX_CSRC_ASCEND_OFFLINE_PARAFORMER_MODEL_ASCEND_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineParaformerModelAscend {
 public:
  ~OfflineParaformerModelAscend();

  explicit OfflineParaformerModelAscend(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineParaformerModelAscend(Manager *mgr, const OfflineModelConfig &config);

  /**
   * @param features A tensor of shape (num_frames, feature_dim)
   *                 before applying LFR.
   * @returns Return a tensor of shape (num_output_frames, encoder_dim)
   */
  std::vector<float> Run(std::vector<float> features) const;

  int32_t VocabSize() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_OFFLINE_PARAFORMER_MODEL_ASCEND_H_
