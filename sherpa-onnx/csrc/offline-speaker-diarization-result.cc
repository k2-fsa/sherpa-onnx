// sherpa-onnx/csrc/offline-speaker-diarization-result.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speaker-diarization-result.h"

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

OfflineSpeakerDiarizationSegment::OfflineSpeakerDiarizationSegment(
    float start, float end, int32_t speaker) {
  if (start > end) {
    SHERPA_ONNX_LOGE("start %.3f should be less than end %.3f", start, end);
    SHERPA_ONNX_EXIT(-1);
  }

  start_ = start;
  end_ = end;
  speaker_ = speaker;
}

}  // namespace sherpa_onnx
