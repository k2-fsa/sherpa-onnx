// sherpa-onnx/csrc/offline-speaker-diarization-result.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speaker-diarization-result.h"

#include <string>

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

std::optional<OfflineSpeakerDiarizationSegment>
OfflineSpeakerDiarizationSegment::Merge(
    const OfflineSpeakerDiarizationSegment &other, float gap) const {
  if (other.speaker_ != speaker_) {
    SHERPA_ONNX_LOGE(
        "The two segments should have the same speaker. this->speaker: %d, "
        "other.speaker: %d",
        speaker_, other.speaker_);
    return std::nullopt;
  }

  if (end_ < other.start_ && end_ + gap >= other.start_) {
    return OfflineSpeakerDiarizationSegment(start_, other.end_, speaker_);
  } else if (other.end_ < start_ && other.end_ + gap >= start_) {
    return OfflineSpeakerDiarizationSegment(other.start_, end_, speaker_);
  } else {
    return std::nullopt;
  }
}

std::string OfflineSpeakerDiarizationSegment::ToString() const {
  char s[128];
  int32_t n = snprintf(s, sizeof(s), "%.3f -- %.3f speaker_%02d", start_, end_,
                       speaker_);

  return {&s[0]};
}

void OfflineSpeakerDiarizationResult::Add(
    const OfflineSpeakerDiarizationSegment &segment) {
  segments_.push_back(segment);
}

int32_t OfflineSpeakerDiarizationResult::NumSegments() const {
  return segments_.size();
}

}  // namespace sherpa_onnx
