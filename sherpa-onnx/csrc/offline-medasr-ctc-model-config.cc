// sherpa-onnx/csrc/offline-medasr-ctc-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-medasr-ctc-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineMedAsrCtcModelConfig::Register(ParseOptions *po) {
  po->Register(
      "medasr", &model,
      "Path to model.onnx from MedASR. Please see "
      "https://github.com/k2-fsa/sherpa-onnx/pull/2934 for available models");
}

bool OfflineMedAsrCtcModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("MedASR model: '%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OfflineMedAsrCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineMedAsrCtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
