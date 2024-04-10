// sherpa-onnx/csrc/audio-tagging-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging-impl.h"

#include "sherpa-onnx/csrc/audio-tagging-zipformer-impl.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

std::unique_ptr<AudioTaggingImpl> AudioTaggingImpl::Create(
    const AudioTaggingConfig &config) {
  if (!config.model.zipformer.model.empty()) {
    return std::make_unique<AudioTaggingZipformerImpl>(config);
  }

  SHERPA_ONNX_LOG(
      "Please specify an audio tagging model! Return a null pointer");
  return nullptr;
}

}  // namespace sherpa_onnx
