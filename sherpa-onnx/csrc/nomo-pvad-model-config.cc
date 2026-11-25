// sherpa-onnx/csrc/nomo-pvad-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/nomo-pvad-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void NomoPvadModelConfig::Register(ParseOptions *po) {
  po->Register("nomo-pvad-model", &model,
               "Path to the NOMO-PVAD ONNX model. Download it from "
               "https://huggingface.co/TuyaNomo/nomo-pvad");

  po->Register("nomo-pvad-speaker-embedding-file", &speaker_embedding_file,
               "Path to a text file containing the 192-d enrollment "
               "embedding of the target speaker as whitespace- or "
               "comma-separated floats");

  po->Register("nomo-pvad-threshold", &threshold,
               "Probability above which a chunk is considered as the target "
               "speaker talking. 0.7-0.85 reduces false acceptance; "
               "0.5-0.6 favors recall");

  po->Register("nomo-pvad-min-silence-duration", &min_silence_duration,
               "In seconds. Wait for this long before ending a speech "
               "segment");

  po->Register("nomo-pvad-min-speech-duration", &min_speech_duration,
               "In seconds. A speech segment shorter than this value is "
               "discarded");

  po->Register("nomo-pvad-max-speech-duration", &max_speech_duration,
               "In seconds. If a speech segment is longer than this value, "
               "then we increase the threshold to 0.9. After finishing "
               "detecting the segment, the threshold is reset");

  po->Register("nomo-pvad-neg-threshold", &neg_threshold,
               "Threshold for speech -> silence. If negative, we use "
               "max(threshold - 0.15, 0.01)");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateNomoPvadModelConfig(const NomoPvadModelConfig &config,
                                        const Exists &exists) {
  if (config.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --nomo-pvad-model");
    return false;
  }

  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("NOMO-PVAD model file '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  if (config.speaker_embedding.empty() &&
      config.speaker_embedding_file.empty()) {
    SHERPA_ONNX_LOGE(
        "Please provide the enrollment embedding of the target speaker, "
        "either with --nomo-pvad-speaker-embedding-file or by setting "
        "NomoPvadModelConfig::speaker_embedding");
    return false;
  }

  if (!config.speaker_embedding_file.empty() &&
      config.speaker_embedding.empty() &&
      !exists(config.speaker_embedding_file)) {
    SHERPA_ONNX_LOGE("NOMO-PVAD speaker embedding file '%s' does not exist",
                     config.speaker_embedding_file.c_str());
    return false;
  }

  if (config.threshold <= 0 || config.threshold >= 1) {
    SHERPA_ONNX_LOGE(
        "Please use a value in (0, 1) for --nomo-pvad-threshold. Given: %f",
        config.threshold);
    return false;
  }

  if (config.min_silence_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --nomo-pvad-min-silence-duration. "
        "Given: %f",
        config.min_silence_duration);
    return false;
  }

  if (config.min_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --nomo-pvad-min-speech-duration. "
        "Given: %f",
        config.min_speech_duration);
    return false;
  }

  if (config.max_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --nomo-pvad-max-speech-duration. "
        "Given: %f",
        config.max_speech_duration);
    return false;
  }

  return true;
}

bool NomoPvadModelConfig::Validate() const {
  return ValidateNomoPvadModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool NomoPvadModelConfig::Validate(Manager *mgr) const {
  return ValidateNomoPvadModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool NomoPvadModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool NomoPvadModelConfig::Validate(NativeResourceManager *mgr) const;
#endif

std::string NomoPvadModelConfig::ToString() const {
  std::ostringstream os;

  os << "NomoPvadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "speaker_embedding_size=" << speaker_embedding.size() << ", ";
  os << "speaker_embedding_file=\"" << speaker_embedding_file << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "max_speech_duration=" << max_speech_duration << ", ";
  os << "neg_threshold=" << neg_threshold << ")";

  return os.str();
}

}  // namespace sherpa_onnx
