// sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model-config.cc
//
// Copyright (c)  2026  Silvio Tomatis
#include "sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSpeakerSegmentationSortformerModelConfig::Register(
    ParseOptions *po) {
  po->Register("sortformer-model", &model,
               "Path to model.onnx of a Sortformer diarization model, e.g., "
               "Nemotron-3-Diarization. If given, no speaker embedding model "
               "is needed.");

  po->Register("sortformer-threshold", &threshold,
               "A frame is assigned to a speaker if the speaker's activity "
               "probability is larger than this value. Valid range: (0, 1).");
}

bool OfflineSpeakerSegmentationSortformerModelConfig::Validate() const {
  if (!(threshold > 0 && threshold < 1)) {
    SHERPA_ONNX_LOGE("--sortformer-threshold must be in (0, 1). Given: %f",
                     threshold);
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Sortformer model: '%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OfflineSpeakerSegmentationSortformerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeakerSegmentationSortformerModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "threshold=" << threshold << ")";

  return os.str();
}

}  // namespace sherpa_onnx
