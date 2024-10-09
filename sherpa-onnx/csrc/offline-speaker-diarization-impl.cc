// sherpa-onnx/csrc/offline-speaker-diarization-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speaker-diarization-impl.h"

#include <memory>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineSpeakerDiarizationImpl>
OfflineSpeakerDiarizationImpl::Create(
    const OfflineSpeakerDiarizationConfig &config) {
  if (!config.segmentation.pyannote.model.empty()) {
    return std::make_unique<OfflineSpeakerDiarizationPyannoteImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please specify a speaker segmentation model.");

  return nullptr;
}

}  // namespace sherpa_onnx
