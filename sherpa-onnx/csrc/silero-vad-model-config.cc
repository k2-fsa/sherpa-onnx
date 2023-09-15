// sherpa-onnx/csrc/silero-vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/silero-vad-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void SilerVadModelConfig::Register(ParseOptions *po) {
  po->Register("silero-vad-model", &model, "Path to silero VAD ONNX model.");

  po->Register("silero-vad-prob", &prob,
               "Speech threshold. Silero VAD outputs speech probabilities for "
               "each audio chunk, probabilities ABOVE this value are "
               "considered as SPEECH. It is better to tune this parameter for "
               "each dataset separately, but lazy "
               "0.5 is pretty good for most datasets.");
  po->Register(
      "silero-vad-min-silence-duration", &min_silence_duration,
      "In seconds.  In the end of each speech chunk wait for "
      "--silero-vad-min-silence-duration seconds before separating it");

  po->Register(
      "silero-vad-window-size", &window_size,
      "In samples. Audio chunks of --silero-vad-window-size samples are fed "
      "to the silero VAD model. WARNING! Silero VAD models were trained using "
      "512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples "
      "for 8000 sample rate. Values other than these may affect model "
      "perfomance!");
}

bool SilerVadModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Silero vad model file %s does not exist", model.c_str());
    return false;
  }

  if (prob < 0.01) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --silero-vad-prob. Given: %f", prob);
    return false;
  }

  if (prob >= 1) {
    SHERPA_ONNX_LOGE(
        "Please use a smaller value for --silero-vad-prob. Given: %f", prob);
    return false;
  }

  return true;
}

std::string SilerVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "SilerVadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "prob=" << prob << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "window_size=" << window_size << ")";

  return os.str();
}

}  // namespace sherpa_onnx
