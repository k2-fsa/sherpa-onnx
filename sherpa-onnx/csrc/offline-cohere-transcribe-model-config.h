// sherpa-onnx/csrc/offline-cohere-transcribe-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_MODEL_CONFIG_H_

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineCohereTranscribeModelConfig {
  std::string encoder;
  std::string decoder;

  // This model supports 14 languages:
  // ar, de, el, en, es, fr, it
  // ja, ko, nl, pl, pt, vi, zh
  std::string language;

  OfflineCohereTranscribeModelConfig() = default;
  OfflineCohereTranscribeModelConfig(const std::string &encoder,
                                     const std::string &decoder,
                                     const std::string &language)
      : encoder(encoder), decoder(decoder), language(language) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_MODEL_CONFIG_H_
