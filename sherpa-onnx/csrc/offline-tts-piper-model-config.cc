// sherpa-onnx/csrc/offline-tts-piper-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-piper-model-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"

namespace sherpa_onnx {

void OfflineTtsPiperModelConfig::Register(ParseOptions *po) {
  po->Register("piper-model", &model, 
               "Path to the Piper .onnx model file");
  
  po->Register("piper-model-config-file", &model_config_file,
               "Path to the Piper model config.json file");
  
  po->Register("piper-data-dir", &data_dir,
               "Path to the espeak-ng-data directory");
}

bool OfflineTtsPiperModelConfig::Validate() const {
  if (!model.empty()) {
    AssertFileExists(model);
  }
  
  if (!model_config_file.empty()) {
    AssertFileExists(model_config_file);
  }

  return true;
}

std::string OfflineTtsPiperModelConfig::ToString() const {
  std::ostringstream os;
  
  os << "OfflineTtsPiperModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "model_config_file=\"" << model_config_file << "\", ";  
  os << "data_dir=\"" << data_dir << "\")";
  
  return os.str();
}

}  // namespace sherpa_onnx