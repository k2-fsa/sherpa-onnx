// sherpa-onnx/csrc/audio-tagging-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging-impl.h"

#include "sherpa-onnx/csrc/audio-tagging-zipformer-impl.h"

namespace sherpa_onnx {

std::unique_ptr<AudioTaggingImpl> AudioTaggingImpl::Create(
    const AudioTaggingConfig &config) {
  return std::make_unique<AudioTaggingZipformerImpl>(config);
}

}  // namespace sherpa_onnx
