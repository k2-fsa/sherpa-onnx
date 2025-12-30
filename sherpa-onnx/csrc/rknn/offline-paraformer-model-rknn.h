// sherpa-onnx/csrc/rknn/offline-paraformer-model-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_RKNN_OFFLINE_PARAFORMER_MODEL_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_OFFLINE_PARAFORMER_MODEL_RKNN_H_

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineParaformerModelRknn {
 public:
  ~OfflineParaformerModelRknn();

  explicit OfflineParaformerModelRknn(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineParaformerModelRknn(Manager *mgr, const OfflineModelConfig &config);

  /**
   * @param features A tensor of shape (num_frames, feature_dim)
   *                 before applying LFR.
   * @returns Return a tensor of shape (num_output_frames, vocab_size)
   */
  std::vector<float> Run(std::vector<float> features) const;

  int32_t VocabSize() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_OFFLINE_PARAFORMER_MODEL_RKNN_H_
