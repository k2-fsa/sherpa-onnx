// sherpa-onnx/csrc/vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/vad-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void VadModelConfig::Register(ParseOptions *po) {
  silero_vad.Register(po);
  ten_vad.Register(po);

  po->Register("vad-sample-rate", &sample_rate,
               "Sample rate expected by the VAD model");

  po->Register("vad-num-threads", &num_threads,
               "Number of threads to run the VAD model");

  po->Register("vad-provider", &provider,
               "Specify a provider to run the VAD model. Supported values: "
               "cpu, cuda, coreml");

  po->Register("vad-debug", &debug,
               "true to display debug information when loading vad models");
}

bool VadModelConfig::Validate() const {
  if (provider != "rknn") {
    if (!silero_vad.model.empty() && EndsWith(silero_vad.model, ".rknn")) {
      SHERPA_ONNX_LOGE(
          "--provider is %s, which is not rknn, but you pass an rknn model "
          "'%s'",
          provider.c_str(), silero_vad.model.c_str());
      return false;
    }
  }

  if (provider == "rknn") {
    if (!silero_vad.model.empty() && EndsWith(silero_vad.model, ".onnx")) {
      SHERPA_ONNX_LOGE("--provider is rknn, but you pass an onnx model '%s'",
                       silero_vad.model.c_str());
      return false;
    }
  }

  if (!silero_vad.model.empty()) {
    return silero_vad.Validate();
  }

  if (!ten_vad.model.empty()) {
    return ten_vad.Validate();
  }

  SHERPA_ONNX_LOGE("Please provide one VAD model.");

  return false;
}

std::string VadModelConfig::ToString() const {
  std::ostringstream os;

  os << "VadModelConfig(";
  os << "silero_vad=" << silero_vad.ToString() << ", ";
  os << "ten_vad=" << ten_vad.ToString() << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_onnx
