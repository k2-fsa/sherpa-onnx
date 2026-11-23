// sherpa-onnx/csrc/vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/vad-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
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
               "cpu, cuda, coreml, openvino");

  po->Register("vad-debug", &debug,
               "true to display debug information when loading vad models");
}

// Provider/model-format consistency checks shared by Validate() and
// Validate(Manager *).
static bool ValidateVadModelProvider(const VadModelConfig &config) {
  if (config.provider != "rknn") {
    if (!config.silero_vad.model.empty() &&
        EndsWith(config.silero_vad.model, ".rknn")) {
      SHERPA_ONNX_LOGE(
          "--provider is %s, which is not rknn, but you pass an rknn model "
          "'%s'",
          config.provider.c_str(), config.silero_vad.model.c_str());
      return false;
    }
  }

  if (config.provider == "rknn") {
    if (!config.silero_vad.model.empty() &&
        EndsWith(config.silero_vad.model, ".onnx")) {
      SHERPA_ONNX_LOGE("--provider is rknn, but you pass an onnx model '%s'",
                       config.silero_vad.model.c_str());
      return false;
    }
  }

  return true;
}

bool VadModelConfig::Validate() const {
  if (!ValidateVadModelProvider(*this)) {
    return false;
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

template <typename Manager>
bool VadModelConfig::Validate(Manager *mgr) const {
  if (!ValidateVadModelProvider(*this)) {
    return false;
  }

  if (!silero_vad.model.empty()) {
    return silero_vad.Validate(mgr);
  }

  if (!ten_vad.model.empty()) {
    return ten_vad.Validate(mgr);
  }

  SHERPA_ONNX_LOGE("Please provide one VAD model.");

  return false;
}

#if __ANDROID_API__ >= 9
template bool VadModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool VadModelConfig::Validate(NativeResourceManager *mgr) const;
#endif

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
