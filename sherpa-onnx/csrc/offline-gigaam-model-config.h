// sherpa-onnx/csrc/offline-gigaam-model-config.h
//
// Copyright (c)  2026  sherpa-onnx contributors

#ifndef SHERPA_ONNX_CSRC_OFFLINE_GIGAAM_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_GIGAAM_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineGigaAMModelConfig {
  std::string encoder;
  std::string advance;
  std::string tokenizer;
  std::string language = "auto";
  std::string head = "e2e";

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

// Slots are ordered as in the Multilingual-v2 release configuration.
int32_t GigaAMLanguageSlot(const std::string &language);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_GIGAAM_MODEL_CONFIG_H_
