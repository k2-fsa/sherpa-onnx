// sherpa-onnx/csrc/offline-dolphin-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-dolphin-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineDolphinModelConfig::Register(ParseOptions *po) {
  po->Register("dolphin-model", &model,
               "Path to model.onnx of Dolphin CTC branch.");
}

bool OfflineDolphinModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Dolphin model '%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OfflineDolphinModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineDolphinModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
