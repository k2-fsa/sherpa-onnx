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
  int32_t lm_num_threads = 1;
  std::string lm_provider = "cpu";

  OnlineLMConfig() = default;

  OnlineLMConfig(const std::string &model, float scale, int32_t lm_num_threads,
                  const std::string &lm_provider)
      : model(model),
        scale(scale),
        lm_num_threads(lm_num_threads),
        lm_provider(lm_provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_LM_CONFIG_H_
