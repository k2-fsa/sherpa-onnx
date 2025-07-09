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
  int32_t lm_num_threads = 1;
  std::string lm_provider = "cpu";

  // LODR
  std::string lodr_fst;
  float lodr_scale = 0.01;
  int32_t lodr_backoff_id = -1;  // -1 means not set

  OfflineLMConfig() = default;

  OfflineLMConfig(const std::string &model, float scale, int32_t lm_num_threads,
                  const std::string &lm_provider, const std::string &lodr_fst,
                  float lodr_scale, int32_t lodr_backoff_id)
      : model(model),
        scale(scale),
        lm_num_threads(lm_num_threads),
        lm_provider(lm_provider),
        lodr_fst(lodr_fst),
        lodr_scale(lodr_scale),
        lodr_backoff_id(lodr_backoff_id) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_LM_CONFIG_H_
