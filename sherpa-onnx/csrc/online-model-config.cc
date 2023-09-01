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

  po->Register("tokens", &tokens, "Path to tokens.txt");

  po->Register("bpe-model", &bpe_model, "Path to bpe.model");

  po->Register("tokens-type", &tokens_type,
               "The tokens type (i.e. the modeling units), supporting bpe, "
               "cjkchar, cjkchar+bpe now.");

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");

  po->Register("model-type", &model_type,
               "Specify it to reduce model initialization time. "
               "Valid values are: conformer, lstm, zipformer, zipformer2."
               "All other values lead to loading the model twice.");
}

bool OnlineModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("tokens: %s does not exist", tokens.c_str());
    return false;
  }

  if (!paraformer.encoder.empty()) {
    return paraformer.Validate();
  }

  return transducer.Validate();
}

std::string OnlineModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineModelConfig(";
  os << "transducer=" << transducer.ToString() << ", ";
  os << "paraformer=" << paraformer.ToString() << ", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "tokens_type=" << tokens_type << ", ";
  os << "bpe_model=\"" << bpe_model << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "model_type=\"" << model_type << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
