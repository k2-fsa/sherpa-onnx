// sherpa-onnx/csrc/online-punctuation-model-config.h
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#ifndef SHERPA_ONNX_CSRC_ONLINE_PUNCTUATION_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_PUNCTUATION_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OnlinePunctuationModelConfig {
  std::string cnn_bilstm;
  std::string bpe_vocab;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  OnlinePunctuationModelConfig() = default;

  OnlinePunctuationModelConfig(const std::string &cnn_bilstm,
                               const std::string &bpe_vocab,
                               int32_t num_threads, bool debug,
                               const std::string &provider)
      : cnn_bilstm(cnn_bilstm),
        bpe_vocab(bpe_vocab),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_PUNCTUATION_MODEL_CONFIG_H_
