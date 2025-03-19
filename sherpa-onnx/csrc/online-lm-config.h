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
  std::string lodr_fst = "";
  float lodr_scale = 0.01;
  int lodr_backoff_id = 0;
  // enable shallow fusion
  bool shallow_fusion = true;

  OnlineLMConfig() = default;

  OnlineLMConfig(const std::string &model, float scale, int32_t lm_num_threads,
                 const std::string &lm_provider, bool shallow_fusion,
                 const std::string &lodr_fst, float lodr_scale)
      : model(model),
        scale(scale),
        lm_num_threads(lm_num_threads),
        lm_provider(lm_provider),
        shallow_fusion(shallow_fusion),
        lodr_fst(lodr_fst),
        lodr_scale(lodr_scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_LM_CONFIG_H_
