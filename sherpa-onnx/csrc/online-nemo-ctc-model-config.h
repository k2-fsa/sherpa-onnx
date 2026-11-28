// sherpa-onnx/csrc/online-nemo-ctc-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_NEMO_CTC_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_NEMO_CTC_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OnlineNeMoCtcModelConfig {
  std::string model;

  OnlineNeMoCtcModelConfig() = default;

  explicit OnlineNeMoCtcModelConfig(const std::string &model) : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  // Like Validate(), for files given by a resource manager (Android asset
  // manager / OHOS native resource manager): paths are relative to the
  // resource root.
  template <typename Manager>
  bool Validate(Manager *mgr) const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_NEMO_CTC_MODEL_CONFIG_H_
