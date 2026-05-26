// sherpa-onnx/csrc/offline-speech-denoiser-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speech-denoiser-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSpeechDenoiserModelConfig::Register(ParseOptions *po) {
  gtcrn.Register(po);
  dpdfnet.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool OfflineSpeechDenoiserModelConfig::Validate() const {
  if (gtcrn.model.empty() && dpdfnet.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide a speech denoising model.");
    return false;
  }

  if (!gtcrn.model.empty()) {
    return gtcrn.Validate();
  }

  return dpdfnet.Validate();
}

std::string OfflineSpeechDenoiserModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserModelConfig(";
  os << "gtcrn=" << gtcrn.ToString() << ", ";
  os << "dpdfnet=" << dpdfnet.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
