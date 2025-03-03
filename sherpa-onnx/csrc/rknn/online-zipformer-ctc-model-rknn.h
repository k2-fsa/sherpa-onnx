// sherpa-onnx/csrc/rknn/online-zipformer-ctc-model-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_RKNN_ONLINE_ZIPFORMER_CTC_MODEL_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_ONLINE_ZIPFORMER_CTC_MODEL_RKNN_H_

#include <memory>
#include <utility>
#include <vector>

#include "rknn_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"

namespace sherpa_onnx {

class OnlineZipformerCtcModelRknn {
 public:
  ~OnlineZipformerCtcModelRknn();

  explicit OnlineZipformerCtcModelRknn(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineZipformerCtcModelRknn(Manager *mgr, const OnlineModelConfig &config);

  std::vector<std::vector<uint8_t>> GetInitStates() const;

  std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>> Run(
      std::vector<float> features,
      std::vector<std::vector<uint8_t>> states) const;

  int32_t ChunkSize() const;

  int32_t ChunkShift() const;

  int32_t VocabSize() const;

  rknn_tensor_attr GetOutAttr() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_ONLINE_ZIPFORMER_CTC_MODEL_RKNN_H_
