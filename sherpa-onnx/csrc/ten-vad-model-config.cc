// sherpa-onnx/csrc/ten-vad-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/ten-vad-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void TenVadModelConfig::Register(ParseOptions *po) {
  po->Register("ten-vad-model", &model, "Path to TEN VAD ONNX model.");

  po->Register("ten-vad-threshold", &threshold,
               "Speech threshold. TEN VAD outputs speech probabilities for "
               "each audio chunk, probabilities ABOVE this value are "
               "considered as SPEECH. It is better to tune this parameter for "
               "each dataset separately, but lazy "
               "0.5 is pretty good for most datasets.");

  po->Register("ten-vad-min-silence-duration", &min_silence_duration,
               "In seconds.  In the end of each speech chunk wait for "
               "--ten-vad-min-silence-duration seconds before separating it");

  po->Register("ten-vad-min-speech-duration", &min_speech_duration,
               "In seconds.  In the end of each silence chunk wait for "
               "--ten-vad-min-speech-duration seconds before separating it");

  po->Register(
      "ten-vad-max-speech-duration", &max_speech_duration,
      "In seconds. If a speech segment is longer than this value, then we "
      "increase the threshold to 0.9. After finishing detecting the segment, "
      "the threshold value is reset to its original value.");

  po->Register(
      "ten-vad-window-size", &window_size,
      "In samples. Audio chunks of --ten-vad-window-size samples are fed "
      "to the ten VAD model. WARNING! Please use 160 or 256 ");
}

bool TenVadModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --ten-vad-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("TEN vad model file '%s' does not exist", model.c_str());
    return false;
  }

  if (threshold < 0.01) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (threshold >= 1) {
    SHERPA_ONNX_LOGE(
        "Please use a smaller value for --ten-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (min_silence_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-min-silence-duration. "
        "Given: "
        "%f",
        min_silence_duration);
    return false;
  }

  if (min_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-min-speech-duration. "
        "Given: "
        "%f",
        min_speech_duration);
    return false;
  }

  if (max_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-max-speech-duration. "
        "Given: "
        "%f",
        max_speech_duration);
    return false;
  }

  return true;
}

std::string TenVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "TenVadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "max_speech_duration=" << max_speech_duration << ", ";
  os << "window_size=" << window_size << ")";

  return os.str();
}

}  // namespace sherpa_onnx
