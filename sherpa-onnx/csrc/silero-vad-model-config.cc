// sherpa-onnx/csrc/silero-vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/silero-vad-model-config.h"

#include <string>

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
      "silero-vad-max-speech-duration", &max_speech_duration,
      "In seconds. If a speech segment is longer than this value, then we "
      "increase the threshold to 0.9. After finishing detecting the segment, "
      "the threshold value is reset to its original value.");

  po->Register(
      "silero-vad-window-size", &window_size,
      "In samples. Audio chunks of --silero-vad-window-size samples are fed "
      "to the silero VAD model. WARNING! Silero VAD models were trained using "
      "512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples "
      "for 8000 sample rate. Values other than these may affect model "
      "performance!");

  po->Register("silero-vad-neg-threshold", &neg_threshold,
               "Negative threshold (noise threshold). If < 0, defaults to "
               "(threshold - 0.15) with lower bound 0.01.");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateSileroVadModelConfig(const SileroVadModelConfig &config,
                                         const Exists &exists) {
  if (config.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --silero-vad-model");
    return false;
  }

  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("Silero vad model file '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  if (config.threshold < 0.01) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --silero-vad-threshold. Given: %f",
        config.threshold);
    return false;
  }

  if (config.threshold >= 1) {
    SHERPA_ONNX_LOGE(
        "Please use a smaller value for --silero-vad-threshold. Given: %f",
        config.threshold);
    return false;
  }

  if (config.min_silence_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --silero-vad-min-silence-duration. "
        "Given: "
        "%f",
        config.min_silence_duration);
    return false;
  }

  if (config.min_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --silero-vad-min-speech-duration. "
        "Given: "
        "%f",
        config.min_speech_duration);
    return false;
  }

  if (config.max_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --silero-vad-max-speech-duration. "
        "Given: "
        "%f",
        config.max_speech_duration);
    return false;
  }

  return true;
}

bool SileroVadModelConfig::Validate() const {
  return ValidateSileroVadModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool SileroVadModelConfig::Validate(Manager *mgr) const {
  return ValidateSileroVadModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool SileroVadModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool SileroVadModelConfig::Validate(NativeResourceManager *mgr) const;
#endif

std::string SileroVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "SileroVadModelConfig(";
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
