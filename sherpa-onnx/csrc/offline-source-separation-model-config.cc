// sherpa-onnx/csrc/offline-source-separation-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-source-separation-model-config.h"

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSourceSeparationModelConfig::Register(ParseOptions *po) {
  spleeter.Register(po);
  uvr.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool OfflineSourceSeparationModelConfig::Validate() const {
  if (!spleeter.vocals.empty()) {
    return spleeter.Validate();
  }

  if (!uvr.model.empty()) {
    return uvr.Validate();
  }

  SHERPA_ONNX_LOGE("Please specify a source separation model");

  return false;
}

std::string OfflineSourceSeparationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSourceSeparationModelConfig(";
  os << "spleeter=" << spleeter.ToString() << ", ";
  os << "uvr=" << uvr.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
