// sherpa-onnx/csrc/offline-source-separation-spleeter-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-source-separation-spleeter-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSourceSeparationSpleeterModelConfig::Register(ParseOptions *po) {
  po->Register("spleeter-vocals", &vocals, "Path to the spleeter vocals model");

  po->Register("spleeter-accompaniment", &accompaniment,
               "Path to the spleeter accompaniment model");
}

bool OfflineSourceSeparationSpleeterModelConfig::Validate() const {
  if (vocals.empty()) {
    SHERPA_ONNX_LOGE("Please provide --spleeter-vocals");
    return false;
  }

  if (!FileExists(vocals)) {
    SHERPA_ONNX_LOGE("spleeter vocals '%s' does not exist. ", vocals.c_str());
    return false;
  }

  if (accompaniment.empty()) {
    SHERPA_ONNX_LOGE("Please provide --spleeter-accompaniment");
    return false;
  }

  if (!FileExists(accompaniment)) {
    SHERPA_ONNX_LOGE("spleeter accompaniment '%s' does not exist. ",
                     accompaniment.c_str());
    return false;
  }

  return true;
}

std::string OfflineSourceSeparationSpleeterModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSourceSeparationSpleeterModelConfig(";
  os << "vocals=\"" << vocals << "\", ";
  os << "accompaniment=\"" << accompaniment << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
