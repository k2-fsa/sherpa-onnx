// sherpa-onnx/csrc/offline-tts-kitten-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-kitten-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineTtsKittenModelConfig::Register(ParseOptions *po) {
  po->Register("kitten-model", &model, "Path to kitten model");
  po->Register("kitten-voices", &voices,
               "Path to voices.bin for kitten models");
  po->Register("kitten-tokens", &tokens,
               "Path to tokens.txt for kitten models");
  po->Register("kitten-lexicon", &lexicon,
               "Path to lexicon.txt for kitten models");
  po->Register("kitten-data-dir", &data_dir,
               "Deprecated. Ignored since v2.0.0: espeak-ng is no longer used. "
               "Use --kitten-lexicon or "
               "GenerationConfig.phoneme_codepoints instead.");
  po->Register("kitten-length-scale", &length_scale,
               "Inverse of speech speed. Larger->Slower; Smaller->faster.");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOfflineTtsKittenModelConfig(
    const OfflineTtsKittenModelConfig &config, const Exists &exists) {
  if (config.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kitten-model");
    return false;
  }

  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("--kitten-model: '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  if (config.voices.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kitten-voices");
    return false;
  }

  if (!exists(config.voices)) {
    SHERPA_ONNX_LOGE("--kitten-voices: '%s' does not exist",
                     config.voices.c_str());
    return false;
  }

  if (config.tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kitten-tokens");
    return false;
  }

  if (!exists(config.tokens)) {
    SHERPA_ONNX_LOGE("--kitten-tokens: '%s' does not exist",
                     config.tokens.c_str());
    return false;
  }

  if (!config.data_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "WARNING: --kitten-data-dir is deprecated and ignored in "
        "sherpa-onnx >= v2.0.0. Please use GenerationConfig.phoneme_codepoints "
        "with an external phonemizer (e.g. piper_phonemize) instead. "
        "See python-api-examples/test-offline-tts-kitten-phonemize.py");
  }

  if (!config.lexicon.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(config.lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!exists(f)) {
        SHERPA_ONNX_LOGE(
            "lexicon '%s' does not exist. Please re-check --kitten-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (config.length_scale <= 0) {
    SHERPA_ONNX_LOGE(
        "Please provide a positive length_scale for --kitten-length-scale. "
        "Given: %.3f",
        config.length_scale);
    return false;
  }

  return true;
}

bool OfflineTtsKittenModelConfig::Validate() const {
  return ValidateOfflineTtsKittenModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineTtsKittenModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineTtsKittenModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OfflineTtsKittenModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineTtsKittenModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineTtsKittenModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsKittenModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "voices=\"" << voices << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "length_scale=" << length_scale << ")";

  return os.str();
}

}  // namespace sherpa_onnx
