// sherpa-onnx/csrc/smart-turn-config.cc
//
// Copyright (c) 2026 Xiaomi Corporation

#include "sherpa-onnx/csrc/smart-turn-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void SmartTurnConfig::Register(ParseOptions *po) {
  po->Register("smart-turn-model", &model, "Path to Smart Turn ONNX model.");
  po->Register("smart-turn-threshold", &threshold,
               "End-of-turn probability threshold.");
  po->Register("smart-turn-sample-rate", &sample_rate,
               "Sample rate expected by the Smart Turn model.");
  po->Register("smart-turn-window-size", &window_size,
               "Audio window size in seconds.");
  po->Register("smart-turn-min-silence-duration", &min_silence_duration,
               "Trailing silence before running Smart Turn, in seconds.");
  po->Register("smart-turn-max-silence-duration", &max_silence_duration,
               "Trailing silence that always ends a turn, in seconds.");
}

bool SmartTurnConfig::Validate() const {
  if (model.empty()) return true;

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Smart Turn model file '%s' does not exist", model.c_str());
    return false;
  }

  if (threshold <= 0 || threshold >= 1 || sample_rate <= 0 || window_size <= 0 ||
      min_silence_duration <= 0 || max_silence_duration < min_silence_duration) {
    SHERPA_ONNX_LOGE("Invalid Smart Turn configuration");
    return false;
  }

  return true;
}

std::string SmartTurnConfig::ToString() const {
  std::ostringstream os;
  os << "SmartTurnConfig(model=\"" << model << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "window_size=" << window_size << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "max_silence_duration=" << max_silence_duration << ")";
  return os.str();
}

}  // namespace sherpa_onnx