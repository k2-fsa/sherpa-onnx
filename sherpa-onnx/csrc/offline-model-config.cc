// sherpa-onnx/csrc/offline-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/offline-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineModelConfig::Register(ParseOptions *po) {
  transducer.Register(po);
  paraformer.Register(po);

  po->Register("tokens", &tokens, "Path to tokens.txt");

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");
}

bool OfflineModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("%s does not exist", tokens.c_str());
    return false;
  }

  if (!paraformer.model.empty()) {
    return paraformer.Validate();
  }

  return transducer.Validate();
}

std::string OfflineModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineModelConfig(";
  os << "transducer=" << transducer.ToString() << ", ";
  os << "paraformer=" << paraformer.ToString() << ", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_onnx
