// sherpa-onnx/csrc/online-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/online-transducer-model-config.h"

namespace sherpa_onnx {

struct OnlineModelConfig {
  OnlineTransducerModelConfig transducer;
  std::string tokens;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  // Valid values:
  //  - conformer, conformer transducer from icefall
  //  - lstm, lstm transducer from icefall
  //  - zipformer, zipformer transducer from icefall
  //  - zipformer2, zipformer2 transducer from icefall
  //
  // All other values are invalid and lead to loading the model twice.
  std::string model_type;

  OnlineModelConfig() = default;
  OnlineModelConfig(const OnlineTransducerModelConfig &transducer,
                    const std::string &tokens, int32_t num_threads, bool debug,
                    const std::string &provider, const std::string &model_type)
      : transducer(transducer),
        tokens(tokens),
        num_threads(num_threads),
        debug(debug),
        provider(provider),
        model_type(model_type) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_
