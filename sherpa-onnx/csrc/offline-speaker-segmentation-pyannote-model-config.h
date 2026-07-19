// sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_CONFIG_H_
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineSpeakerSegmentationPyannoteModelConfig {
  std::string model;
  float window_shift_ratio = 0.1f;

  OfflineSpeakerSegmentationPyannoteModelConfig() = default;

  explicit OfflineSpeakerSegmentationPyannoteModelConfig(
      const std::string &model, float window_shift_ratio = 0.1f)
      : model(model), window_shift_ratio(window_shift_ratio) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_CONFIG_H_
