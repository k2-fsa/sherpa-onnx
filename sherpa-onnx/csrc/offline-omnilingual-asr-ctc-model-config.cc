// sherpa-onnx/csrc/offline-omnilingual-asr-ctc-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-omnilingual-asr-ctc-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineOmnilingualAsrCtcModelConfig::Register(ParseOptions *po) {
  po->Register("omnilingual-asr-model", &model,
               "Path to Omnilingual ASR CTC model");
}

bool OfflineOmnilingualAsrCtcModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Omnilingual ASR CTC model file '%s' does not exist",
                     model.c_str());
    return false;
  }

  return true;
}

std::string OfflineOmnilingualAsrCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineOmnilingualAsrCtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
