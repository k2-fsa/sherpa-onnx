// sherpa-onnx/csrc/online-t-one-ctc-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-t-one-ctc-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineToneCtcModelConfig::Register(ParseOptions *po) {
  po->Register("t-one-ctc-model", &model,
               "Path to CTC model.onnx from T-one. Please see "
               "https://github.com/k2-fsa/sherpa-onnx/pull/2571");
}

bool OnlineToneCtcModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("T-one CTC model '%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OnlineToneCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineToneCtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
