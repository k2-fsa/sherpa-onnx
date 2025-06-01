// sherpa-onnx/csrc/offline-source-separation-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/offline-source-separation-spleeter-model-config.h"
#include "sherpa-onnx/csrc/offline-source-separation-uvr-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineSourceSeparationModelConfig {
  OfflineSourceSeparationSpleeterModelConfig spleeter;
  OfflineSourceSeparationUvrModelConfig uvr;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  OfflineSourceSeparationModelConfig() = default;

  OfflineSourceSeparationModelConfig(
      const OfflineSourceSeparationSpleeterModelConfig &spleeter,
      const OfflineSourceSeparationUvrModelConfig &uvr, int32_t num_threads,
      bool debug, const std::string &provider)
      : spleeter(spleeter),
        uvr(uvr),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_MODEL_CONFIG_H_
