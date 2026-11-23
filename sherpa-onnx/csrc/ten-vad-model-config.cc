// sherpa-onnx/csrc/ten-vad-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/ten-vad-model-config.h"

#include <string>

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

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateTenVadModelConfig(const TenVadModelConfig &config,
                                      const Exists &exists) {
  if (config.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --ten-vad-model");
    return false;
  }

  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("TEN vad model file '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  if (config.threshold < 0.01) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-threshold. Given: %f",
        config.threshold);
    return false;
  }

  if (config.threshold >= 1) {
    SHERPA_ONNX_LOGE(
        "Please use a smaller value for --ten-vad-threshold. Given: %f",
        config.threshold);
    return false;
  }

  if (config.min_silence_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-min-silence-duration. "
        "Given: "
        "%f",
        config.min_silence_duration);
    return false;
  }

  if (config.min_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-min-speech-duration. "
        "Given: "
        "%f",
        config.min_speech_duration);
    return false;
  }

  if (config.max_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-max-speech-duration. "
        "Given: "
        "%f",
        config.max_speech_duration);
    return false;
  }

  return true;
}

bool TenVadModelConfig::Validate() const {
  return ValidateTenVadModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool TenVadModelConfig::Validate(Manager *mgr) const {
  return ValidateTenVadModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool TenVadModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool TenVadModelConfig::Validate(NativeResourceManager *mgr) const;
#endif

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
