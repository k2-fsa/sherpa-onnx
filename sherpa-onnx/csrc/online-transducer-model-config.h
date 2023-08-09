// sherpa-onnx/csrc/online-transducer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OnlineTransducerModelConfig {
  std::string encoder;
  std::string decoder;
  std::string joiner;

  OnlineTransducerModelConfig() = default;
  OnlineTransducerModelConfig(const std::string &encoder,
                              const std::string &decoder,
                              const std::string &joiner)
      : encoder(encoder), decoder(decoder), joiner(joiner) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_
