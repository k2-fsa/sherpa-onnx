// sherpa-onnx/csrc/offline-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speaker-diarization.h"

#include "sherpa-onnx/csrc/offline-speaker-diarization-impl.h"

namespace sherpa_onnx {

OfflineSpeakerDiarization::OfflineSpeakerDiarization(
    const OfflineSpeakerDiarizationConfig &config) {}

OfflineSpeakerDiarization::~OfflineSpeakerDiarization() = default;

OfflineSpeakerDiarizationResult OfflineSpeakerDiarization::Process(
    const float *audio, int32_t n,
    OfflineSpeakerDiarizationProgressCallback callback /*= nullptr*/) const {
  return impl_->Process(audio, n, callback);
}

}  // namespace sherpa_onnx
