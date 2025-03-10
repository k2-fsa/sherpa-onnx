// sherpa-onnx/csrc/offline-speech-denoiser-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speech-denoiser-model-config.h"

#include <string>

namespace sherpa_onnx {

void OfflineSpeechDenoiserModelConfig::Register(ParseOptions *po) {}

bool OfflineSpeechDenoiserModelConfig::Validate() const { return true; }

std::string OfflineSpeechDenoiserModelConfig::ToString() const { return {}; }

}  // namespace sherpa_onnx
