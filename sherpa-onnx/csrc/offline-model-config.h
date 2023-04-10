// sherpa-onnx/csrc/offline-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/offline-nemo-enc-dec-ctc-model-config.h"
#include "sherpa-onnx/csrc/offline-paraformer-model-config.h"
#include "sherpa-onnx/csrc/offline-transducer-model-config.h"

namespace sherpa_onnx {

struct OfflineModelConfig {
  OfflineTransducerModelConfig transducer;
  OfflineParaformerModelConfig paraformer;
  OfflineNemoEncDecCtcModelConfig nemo_ctc;

  std::string tokens;
  int32_t num_threads = 2;
  bool debug = false;

  OfflineModelConfig() = default;
  OfflineModelConfig(const OfflineTransducerModelConfig &transducer,
                     const OfflineParaformerModelConfig &paraformer,
                     const OfflineNemoEncDecCtcModelConfig &nemo_ctc,
                     const std::string &tokens, int32_t num_threads, bool debug)
      : transducer(transducer),
        paraformer(paraformer),
        nemo_ctc(nemo_ctc),
        tokens(tokens),
        num_threads(num_threads),
        debug(debug) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MODEL_CONFIG_H_
