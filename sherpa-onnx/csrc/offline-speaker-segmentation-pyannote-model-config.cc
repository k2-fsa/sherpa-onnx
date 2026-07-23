// sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSpeakerSegmentationPyannoteModelConfig::Register(ParseOptions *po) {
  po->Register("pyannote-model", &model,
               "Path to model.onnx of the Pyannote segmentation model.");
  po->Register(
      "pyannote-window-shift-ratio", &window_shift_ratio,
      "Window shift as a ratio of the Pyannote segmentation window size. "
      "Default: 0.1. Valid range: 0 < ratio <= 1. Smaller values mean more "
      "overlap, higher quality, and slower processing.");
}

bool OfflineSpeakerSegmentationPyannoteModelConfig::Validate() const {
  if (!(window_shift_ratio > 0 && window_shift_ratio <= 1)) {
    SHERPA_ONNX_LOGE(
        "--pyannote-window-shift-ratio must be in (0, 1]. Given: %f",
        window_shift_ratio);
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Pyannote segmentation model: '%s' does not exist",
                     model.c_str());
    return false;
  }

  return true;
}

std::string OfflineSpeakerSegmentationPyannoteModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeakerSegmentationPyannoteModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "window_shift_ratio=" << window_shift_ratio << ")";

  return os.str();
}

}  // namespace sherpa_onnx
