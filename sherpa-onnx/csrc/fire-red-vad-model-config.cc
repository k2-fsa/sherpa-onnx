// sherpa-onnx/csrc/fire-red-vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/fire-red-vad-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void FireRedVadModelConfig::Register(ParseOptions *po) {
  po->Register("fire-red-vad-model", &model, "Path to FireRed VAD ONNX model.");

  po->Register("fire-red-vad-threshold", &threshold,
               "Speech threshold. FireRed VAD outputs speech probabilities for "
               "each audio chunk, probabilities ABOVE this value are "
               "considered as SPEECH. It is better to tune this parameter for "
               "each dataset separately, but lazy "
               "0.5 is pretty good for most datasets.");

  po->Register(
      "fire-red-vad-min-silence-duration", &min_silence_duration,
      "In seconds.  In the end of each speech chunk wait for "
      "--fire-red-vad-min-silence-duration seconds before separating it");

  po->Register(
      "fire-red-vad-min-speech-duration", &min_speech_duration,
      "In seconds.  In the end of each silence chunk wait for "
      "--fire-red-vad-min-speech-duration seconds before separating it");

  po->Register(
      "fire-red-vad-max-speech-duration", &max_speech_duration,
      "In seconds. If a speech segment is longer than this value, then we "
      "increase the threshold to 0.9. After finishing detecting the segment, "
      "the threshold value is reset to its original value.");

  po->Register(
      "fire-red-vad-window-size", &window_size,
      "In samples. Audio chunks of --fire-red-vad-window-size samples are fed "
      "to the fire-red VAD model. WARNING! FireRed VAD models were trained "
      "using "
      "512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples "
      "for 8000 sample rate. Values other than these may affect model "
      "performance!");

  po->Register("fire-red-vad-neg-threshold", &neg_threshold,
               "Negative threshold (noise threshold). If < 0, defaults to "
               "(threshold - 0.15) with lower bound 0.01.");
}

bool FireRedVadModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --fire-red-vad-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("FireRed vad model file '%s' does not exist",
                     model.c_str());
    return false;
  }

  if (threshold < 0.01) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --fire-red-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (threshold >= 1) {
    SHERPA_ONNX_LOGE(
        "Please use a smaller value for --fire-red-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (min_silence_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --fire-red-vad-min-silence-duration. "
        "Given: "
        "%f",
        min_silence_duration);
    return false;
  }

  if (min_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --fire-red-vad-min-speech-duration. "
        "Given: "
        "%f",
        min_speech_duration);
    return false;
  }

  if (max_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --fire-red-vad-max-speech-duration. "
        "Given: "
        "%f",
        max_speech_duration);
    return false;
  }

  return true;
}

std::string FireRedVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "FireRedVadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "max_speech_duration=" << max_speech_duration << ", ";
  os << "window_size=" << window_size << ", ";
  os << "neg_threshold=" << neg_threshold << ")";

  return os.str();
}

}  // namespace sherpa_onnx
