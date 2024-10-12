// sherpa-onnx/csrc/offline-speaker-diarization.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_H_

#include <functional>
#include <memory>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/fast-clustering-config.h"
#include "sherpa-onnx/csrc/offline-speaker-diarization-result.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

struct OfflineSpeakerDiarizationConfig {
  OfflineSpeakerSegmentationModelConfig segmentation;
  SpeakerEmbeddingExtractorConfig embedding;
  FastClusteringConfig clustering;

  // if a segment is less than this value, then it is discarded
  float min_duration_on = 0.3;  // in seconds

  // if the gap between to segments of the same speaker is less than this value,
  // then these two segments are merged into a single segment.
  // We do this recursively.
  float min_duration_off = 0.5;  // in seconds

  OfflineSpeakerDiarizationConfig() = default;

  OfflineSpeakerDiarizationConfig(
      const OfflineSpeakerSegmentationModelConfig &segmentation,
      const SpeakerEmbeddingExtractorConfig &embedding,
      const FastClusteringConfig &clustering, float min_duration_on,
      float min_duration_off)
      : segmentation(segmentation),
        embedding(embedding),
        clustering(clustering),
        min_duration_on(min_duration_on),
        min_duration_off(min_duration_off) {}

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

#if __ANDROID_API__ >= 9
  OfflineSpeakerDiarization(AAssetManager *mgr,
                            const OfflineSpeakerDiarizationConfig &config);
#endif

  ~OfflineSpeakerDiarization();

  // Expected sample rate of the input audio samples
  int32_t SampleRate() const;

  // Note: Only config.clustering is used. All other fields in config are
  // ignored
  void SetConfig(const OfflineSpeakerDiarizationConfig &config);

  OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const;

 private:
  std::unique_ptr<OfflineSpeakerDiarizationImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_H_
