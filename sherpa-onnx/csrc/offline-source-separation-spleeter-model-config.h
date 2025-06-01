// sherpa-onnx/csrc/offline-source-separation-spleeter-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/offline-source-separation-spleeter-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineSourceSeparationSpleeterModelConfig {
  std::string vocals;

  std::string accompaniment;

  OfflineSourceSeparationSpleeterModelConfig() = default;

  OfflineSourceSeparationSpleeterModelConfig(const std::string &vocals,
                                             const std::string &accompaniment)
      : vocals(vocals), accompaniment(accompaniment) {}

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_MODEL_CONFIG_H_
