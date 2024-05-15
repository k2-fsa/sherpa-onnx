// sherpa-onnx/csrc/online-nemo-ctc-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-nemo-ctc-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineNeMoCtcModelConfig::Register(ParseOptions *po) {
  po->Register("nemo-ctc-model", &model,
               "Path to CTC model.onnx from NeMo. Please see "
               "https://github.com/k2-fsa/sherpa-onnx/pull/843");
}

bool OnlineNeMoCtcModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("NeMo CTC model '%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OnlineNeMoCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineNeMoCtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
