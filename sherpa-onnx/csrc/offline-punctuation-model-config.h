// sherpa-onnx/csrc/offline-punctuation-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflinePunctuationModelConfig {
  std::string ct_transformer;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";
  bool enable_cpu_mem_arena = true;
  bool enable_mem_pattern = true;

  OfflinePunctuationModelConfig() = default;

  OfflinePunctuationModelConfig(const std::string &ct_transformer,
                                int32_t num_threads, bool debug,
                                const std::string &provider,
                                bool enable_cpu_mem_arena = true,
                                bool enable_mem_pattern = true)
      : ct_transformer(ct_transformer),
        num_threads(num_threads),
        debug(debug),
        provider(provider),
        enable_cpu_mem_arena(enable_cpu_mem_arena),
        enable_mem_pattern(enable_mem_pattern) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_MODEL_CONFIG_H_
