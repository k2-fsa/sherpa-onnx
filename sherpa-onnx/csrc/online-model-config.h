// sherpa-onnx/csrc/online-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/online-paraformer-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"
#include "sherpa-onnx/csrc/online-wenet-ctc-model-config.h"
#include "sherpa-onnx/csrc/online-zipformer2-ctc-model-config.h"

namespace sherpa_onnx {

struct OnlineModelConfig {
  OnlineTransducerModelConfig transducer;
  OnlineParaformerModelConfig paraformer;
  OnlineWenetCtcModelConfig wenet_ctc;
  OnlineZipformer2CtcModelConfig zipformer2_ctc;
  std::string tokens;
  int32_t num_threads = 1;
  int32_t warm_up = 0;
  bool debug = false;
  std::string provider = "cpu";

  // Valid values:
  //  - conformer, conformer transducer from icefall
  //  - lstm, lstm transducer from icefall
  //  - zipformer, zipformer transducer from icefall
  //  - zipformer2, zipformer2 transducer or CTC from icefall
  //  - wenet_ctc, wenet CTC model
  //
  // All other values are invalid and lead to loading the model twice.
  std::string model_type;

  OnlineModelConfig() = default;
  OnlineModelConfig(const OnlineTransducerModelConfig &transducer,
                    const OnlineParaformerModelConfig &paraformer,
                    const OnlineWenetCtcModelConfig &wenet_ctc,
                    const OnlineZipformer2CtcModelConfig &zipformer2_ctc,
                    const std::string &tokens, int32_t num_threads,
                    int32_t warm_up, bool debug, const std::string &provider,
                    const std::string &model_type)
      : transducer(transducer),
        paraformer(paraformer),
        wenet_ctc(wenet_ctc),
        zipformer2_ctc(zipformer2_ctc),
        tokens(tokens),
        num_threads(num_threads),
        warm_up(warm_up),
        debug(debug),
        provider(provider),
        model_type(model_type) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_
