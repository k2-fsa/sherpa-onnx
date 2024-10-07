// sherpa-onnx/csrc/offline-speaker-diarization.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_H_

#include <functional>
#include <memory>
#include <string>

#include "sherpa-onnx/csrc/offline-speaker-diarization-result.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

struct OfflineSpeakerDiarizationConfig {
  OfflineSpeakerSegmentationModelConfig segmentation;
  SpeakerEmbeddingExtractorConfig embedding;

  OfflineSpeakerDiarizationConfig() = default;
  OfflineSpeakerDiarizationConfig(
      const OfflineSpeakerSegmentationModelConfig &segmentation,
      const SpeakerEmbeddingExtractorConfig &embedding);

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

class OfflineSpeakerDiarizationImpl;

using OfflineSpeakerDiarizationProgressCallback = std::function<int32_t(
    int32_t processed_chunks, int32_t num_chunks, void *arg)>;

class OfflineSpeakerDiarization {
 public:
  explicit OfflineSpeakerDiarization(
      const OfflineSpeakerDiarizationConfig &config);

  ~OfflineSpeakerDiarization();

  OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr) const;

 private:
  std::unique_ptr<OfflineSpeakerDiarizationImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_H_
