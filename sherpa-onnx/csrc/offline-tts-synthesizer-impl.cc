// sherpa-onnx/csrc/offline-tts-synthesizer-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-synthesizer-impl.h"

#include "sherpa-onnx/csrc/offline-tts-synthesizer-vits-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineTtsSynthesizerImpl> OfflineTtsSynthesizerImpl::Create(
    const OfflineTtsSynthesizerConfig &config) {
  // TODO(fangjun): Support other types
  return std::make_unique<OfflineTtsSynthesizerVitsImpl>(config);
}

}  // namespace sherpa_onnx
