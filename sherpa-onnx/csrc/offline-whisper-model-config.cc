// sherpa-onnx/csrc/offline-whisper-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineWhisperModelConfig::Register(ParseOptions *po) {
  po->Register("whisper-encoder", &encoder,
               "Path to onnx encoder of whisper, e.g., tiny-encoder.onnx, "
               "medium.en-encoder.onnx.");

  po->Register("whisper-decoder", &decoder,
               "Path to onnx decoder of whisper, e.g., tiny-decoder.onnx, "
               "medium.en-decoder.onnx.");
}

bool OfflineWhisperModelConfig::Validate() const {
  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("whisper encoder file %s does not exist", encoder.c_str());
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("whisper decoder file %s does not exist", decoder.c_str());
    return false;
  }

  return true;
}

std::string OfflineWhisperModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineWhisperModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
