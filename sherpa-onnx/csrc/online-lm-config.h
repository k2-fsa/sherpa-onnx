// sherpa-onnx/csrc/online-lm-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_LM_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_LM_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OnlineLMConfig {
  // path to the onnx model
  std::string model;

  // LM scale
  float scale = 0.5;

  OnlineLMConfig() = default;

  OnlineLMConfig(const std::string &model, float scale)
      : model(model), scale(scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_LM_CONFIG_H_
