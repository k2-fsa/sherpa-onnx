// sherpa-onnx/csrc/offline-source-separation-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-source-separation-model-config.h"

namespace sherpa_onnx {

void OfflineSourceSeparationModelConfig::Register(ParseOptions *po) {
  spleeter.Register(po);
}

bool OfflineSourceSeparationModelConfig::Validate() const {
  return spleeter.Validate();
}

std::string OfflineSourceSeparationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSourceSeparationModelConfig(";
  os << "spleeter=" << spleeter.ToString() << ")";

  return os.str();
}

}  // namespace sherpa_onnx
