// sherpa-onnx/csrc/offline-source-separation-uvr-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/offline-source-separation-uvr-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineSourceSeparationUvrModelConfig {
  std::string model;

  OfflineSourceSeparationUvrModelConfig() = default;

  explicit OfflineSourceSeparationUvrModelConfig(const std::string &model)
      : model(model) {}

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_MODEL_CONFIG_H_
