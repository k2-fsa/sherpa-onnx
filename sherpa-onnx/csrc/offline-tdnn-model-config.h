// sherpa-onnx/csrc/offline-tdnn-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TDNN_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TDNN_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// for https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR/tdnn
struct OfflineTdnnModelConfig {
  std::string model;

  OfflineTdnnModelConfig() = default;
  explicit OfflineTdnnModelConfig(const std::string &model) : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TDNN_MODEL_CONFIG_H_
