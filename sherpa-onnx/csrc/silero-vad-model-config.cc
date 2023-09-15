// sherpa-onnx/csrc/silero-vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/silero-vad-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void SileroVadModelConfig::Register(ParseOptions *po) {
  po->Register("silero-vad-model", &model, "Path to silero VAD ONNX model.");

  po->Register("silero-vad-threshold", &threshold,
               "Speech threshold. Silero VAD outputs speech probabilities for "
               "each audio chunk, probabilities ABOVE this value are "
               "considered as SPEECH. It is better to tune this parameter for "
               "each dataset separately, but lazy "
               "0.5 is pretty good for most datasets.");

  po->Register(
      "silero-vad-min-silence-duration", &min_silence_duration,
      "In seconds.  In the end of each speech chunk wait for "
      "--silero-vad-min-silence-duration seconds before separating it");

  po->Register("silero-vad-min-speech-duration", &min_speech_duration,
               "In seconds.  In the end of each silence chunk wait for "
               "--silero-vad-min-speech-duration seconds before separating it");

  po->Register(
      "silero-vad-window-size", &window_size,
      "In samples. Audio chunks of --silero-vad-window-size samples are fed "
      "to the silero VAD model. WARNING! Silero VAD models were trained using "
      "512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples "
      "for 8000 sample rate. Values other than these may affect model "
      "perfomance!");
}

bool SileroVadModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --silero-vad-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Silero vad model file %s does not exist", model.c_str());
    return false;
  }

  if (threshold < 0.01) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --silero-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (threshold >= 1) {
    SHERPA_ONNX_LOGE(
        "Please use a smaller value for --silero-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  return true;
}

std::string SileroVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "SilerVadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "window_size=" << window_size << ")";

  return os.str();
}

}  // namespace sherpa_onnx
