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
  nemo_ctc.Register(po);
  whisper.Register(po);

  po->Register("tokens", &tokens, "Path to tokens.txt");

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");

  po->Register("model-type", &model_type,
               "Specify it to reduce model initialization time. "
               "Valid values are: transducer, paraformer, nemo_ctc, whisper."
               "All other values lead to loading the model twice.");
}

bool OfflineModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("tokens: %s does not exist", tokens.c_str());
    return false;
  }

  if (!paraformer.model.empty()) {
    return paraformer.Validate();
  }

  if (!nemo_ctc.model.empty()) {
    return nemo_ctc.Validate();
  }

  if (!whisper.encoder.empty()) {
    return whisper.Validate();
  }

  return transducer.Validate();
}

std::string OfflineModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineModelConfig(";
  os << "transducer=" << transducer.ToString() << ", ";
  os << "paraformer=" << paraformer.ToString() << ", ";
  os << "nemo_ctc=" << nemo_ctc.ToString() << ", ";
  os << "whisper=" << whisper.ToString() << ", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "model_type=\"" << model_type << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
