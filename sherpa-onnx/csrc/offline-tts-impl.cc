// sherpa-onnx/csrc/offline-tts-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-impl.h"

#include <memory>

#include "sherpa-onnx/csrc/offline-tts-vits-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    const OfflineTtsConfig &config) {
  // TODO(fangjun): Support other types
  return std::make_unique<OfflineTtsVitsImpl>(config);
}

}  // namespace sherpa_onnx
