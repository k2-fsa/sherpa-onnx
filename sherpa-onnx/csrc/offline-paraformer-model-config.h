// sherpa-onnx/csrc/offline-paraformer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineParaformerModelConfig {
  // for ascend npu,
  // model is "/path/to/encoder.om,/path/to/predictor.om,/path/to/decoder.om"
  std::string model;

  OfflineParaformerModelConfig() = default;
  explicit OfflineParaformerModelConfig(const std::string &model)
      : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODEL_CONFIG_H_
