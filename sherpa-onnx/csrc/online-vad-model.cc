// sherpa-onnx/csrc/online-vad-model.cc
//
// Copyright (c)  2026  Manickavela Arumugam

#include "sherpa-onnx/csrc/online-vad-model.h"

#include <memory>

#include "sherpa-onnx/csrc/online-silero-vad.h"

namespace sherpa_onnx {

std::unique_ptr<OnlineVadModel> OnlineVadModel::Create(
    const OnlineVadModelConfig &config) {
  if (!config.IsEnabled()) {
    return nullptr;
  }
  // Silero is the only online VAD model on this branch; new models plug in
  // here, selected from the config.
  return std::make_unique<OnlineSileroVad>(config);
}

}  // namespace sherpa_onnx
