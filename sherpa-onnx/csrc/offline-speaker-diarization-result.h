// sherpa-onnx/csrc/offline-speaker-diarization-result.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_RESULT_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_RESULT_H_

#include <cstdint>
#include <optional>
#include <vector>

namespace sherpa_onnx {

class OfflineSpeakerDiarizationSegment {
 public:
  OfflineSpeakerDiarizationSegment(float start, float end, int32_t speaker);

  // If the gap between the two segments is less than the given gap, then we
  // merge them and return a new segment. Otherwise, it returns null.
  std::optional<OfflineSpeakerDiarizationSegment> Merge(
      const OfflineSpeakerDiarizationSegment &other, float gap) const;

 private:
  float start_;      // in seconds
  float end_;        // in seconds
  int32_t speaker_;  // ID of the speaker, starting from 0
};

class OfflineSpeakerDiarizationResult {
 public:
  // Add a new segment
  void Add(const OfflineSpeakerDiarizationSegment &segment);

  // Number of distinct speakers contained in this object at this point
  int32_t NumSpeakers() const;

  // Return a list of segments sorted by segment.start time
  std::vector<OfflineSpeakerDiarizationSegment> SortByStartTime() const;

  // ans.size() == NumSpeakers().
  // ans[i] is for speaker_i and is sorted by start time
  std::vector<std::vector<OfflineSpeakerDiarizationSegment>> SortBySpeaker()
      const;

 private:
  std::vector<OfflineSpeakerDiarizationSegment> segments_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_RESULT_H_
