// sherpa-onnx/csrc/offline-lm-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_LM_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_LM_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineLMConfig {
  // path to the onnx model
  std::string model;

  // LM scale
  float scale = 0.5;

  OfflineLMConfig() = default;

  OfflineLMConfig(const std::string &model, float scale)
      : model(model), scale(scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_LM_CONFIG_H_
