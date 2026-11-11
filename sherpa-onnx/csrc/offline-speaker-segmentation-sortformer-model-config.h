// sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model-config.h
//
// Copyright (c)  2026  Silvio Tomatis

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_SORTFORMER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_SORTFORMER_MODEL_CONFIG_H_
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// Config for end-to-end diarization models from NVIDIA NeMo based on the
// streaming Sortformer, e.g., Nemotron-3-Diarization. They do not need a
// speaker embedding model or clustering.
struct OfflineSpeakerSegmentationSortformerModelConfig {
  std::string model;

  // A frame is assigned to a speaker if the speaker's activity probability
  // is larger than this value
  float threshold = 0.5f;

  OfflineSpeakerSegmentationSortformerModelConfig() = default;

  explicit OfflineSpeakerSegmentationSortformerModelConfig(
      const std::string &model, float threshold = 0.5f)
      : model(model), threshold(threshold) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_SORTFORMER_MODEL_CONFIG_H_
