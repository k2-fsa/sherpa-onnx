// sherpa-onnx/csrc/offline-dolphin-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineDolphinModelConfig {
  std::string model;

  OfflineDolphinModelConfig() = default;
  explicit OfflineDolphinModelConfig(const std::string &model) : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_MODEL_CONFIG_H_
