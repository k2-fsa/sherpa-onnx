// sherpa-onnx/csrc/offline-w2v-bert-ctc-model-config.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-w2v-bert-ctc-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineW2vBertCtcModelConfig::Register(ParseOptions *po) {
  po->Register("w2v-bert-ctc", &model,
               "Path to model.onnx of a w2v-BERT 2.0 CTC model. Please see "
               "scripts/w2v-bert/README.md for how to export one");
}

bool OfflineW2vBertCtcModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("w2v-BERT CTC model: '%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OfflineW2vBertCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineW2vBertCtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
