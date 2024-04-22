// sherpa-onnx/csrc/online-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineModelConfig::Register(ParseOptions *po) {
  transducer.Register(po);
  paraformer.Register(po);
  wenet_ctc.Register(po);
  zipformer2_ctc.Register(po);

  po->Register("tokens", &tokens, "Path to tokens.txt");

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("warm-up", &warm_up,
               "Number of warm-up to run the onnxruntime"
               "Valid vales are: zipformer2");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");

  po->Register(
      "model-type", &model_type,
      "Specify it to reduce model initialization time. "
      "Valid values are: conformer, lstm, zipformer, zipformer2, wenet_ctc"
      "All other values lead to loading the model twice.");
}

bool OnlineModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (!paraformer.encoder.empty()) {
    return paraformer.Validate();
  }

  if (!wenet_ctc.model.empty()) {
    return wenet_ctc.Validate();
  }

  if (!zipformer2_ctc.model.empty()) {
    return zipformer2_ctc.Validate();
  }

  return transducer.Validate();
}

std::string OnlineModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineModelConfig(";
  os << "transducer=" << transducer.ToString() << ", ";
  os << "paraformer=" << paraformer.ToString() << ", ";
  os << "wenet_ctc=" << wenet_ctc.ToString() << ", ";
  os << "zipformer2_ctc=" << zipformer2_ctc.ToString() << ", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "warm_up=" << warm_up << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "model_type=\"" << model_type << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
