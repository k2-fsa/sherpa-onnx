// sherpa-onnx/csrc/offline-lm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-lm.h"

#include "sherpa-onnx/csrc/offline-rnn-lm.h"
namespace sherpa_onnx {

std::unique_ptr<OfflineLM> OfflineLM::Create(const OfflineLMConfig &config) {
  return std::make_unique<OfflineRnnLM>(config);
}

}  // namespace sherpa_onnx
