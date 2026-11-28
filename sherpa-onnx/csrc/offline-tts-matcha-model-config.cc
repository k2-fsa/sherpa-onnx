// sherpa-onnx/csrc/offline-tts-matcha-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-matcha-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineTtsMatchaModelConfig::Register(ParseOptions *po) {
  po->Register("matcha-acoustic-model", &acoustic_model,
               "Path to matcha acoustic model");
  po->Register("matcha-vocoder", &vocoder, "Path to matcha vocoder");
  po->Register(
      "matcha-lexicon", &lexicon,
      "Path to lexicon.txt for Matcha models. You can pass multiple "
      "files separated by comma , e.g., lexicon.txt,lexicon2.txt,lexicon3.txt");
  po->Register("matcha-tokens", &tokens,
               "Path to tokens.txt for Matcha models");
  po->Register("matcha-data-dir", &data_dir,
               "Deprecated. Ignored since v2.0.0: espeak-ng is no longer used. "
               "Use --matcha-lexicon or "
               "GenerationConfig.phoneme_codepoints instead.");
  po->Register("matcha-dict-dir", &dict_dir,
               "Not used. You don't need to provide a value for it");
  po->Register("matcha-noise-scale", &noise_scale,
               "noise_scale for Matcha models");
  po->Register("matcha-length-scale", &length_scale,
               "Speech speed. Larger->Slower; Smaller->faster.");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOfflineTtsMatchaModelConfig(
    const OfflineTtsMatchaModelConfig &config, const Exists &exists) {
  if (config.acoustic_model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --matcha-acoustic-model");
    return false;
  }

  if (!exists(config.acoustic_model)) {
    SHERPA_ONNX_LOGE("--matcha-acoustic-model: '%s' does not exist",
                     config.acoustic_model.c_str());
    return false;
  }

  if (config.tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --matcha-tokens");
    return false;
  }

  if (!exists(config.tokens)) {
    SHERPA_ONNX_LOGE("--matcha-tokens: '%s' does not exist",
                     config.tokens.c_str());
    return false;
  }

  if (!config.data_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "WARNING: --matcha-data-dir is deprecated. Please use "
        "--matcha-lexicon or GenerationConfig.phoneme_codepoints with an "
        "external phonemizer (e.g. piper_phonemize) instead.");
  }

  if (!config.lexicon.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(config.lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!exists(f)) {
        SHERPA_ONNX_LOGE(
            "lexicon '%s' does not exist. Please re-check --matcha-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (!config.dict_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "From sherpa-onnx v1.12.15, you don't need to provide dict_dir for "
        "this model. Ignore it");
  }

  return true;
}

bool OfflineTtsMatchaModelConfig::Validate() const {
  return ValidateOfflineTtsMatchaModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineTtsMatchaModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineTtsMatchaModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OfflineTtsMatchaModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineTtsMatchaModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineTtsMatchaModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsMatchaModelConfig(";
  os << "acoustic_model=\"" << acoustic_model << "\", ";
  os << "vocoder=\"" << vocoder << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "noise_scale=" << noise_scale << ", ";
  os << "length_scale=" << length_scale << ")";

  return os.str();
}

}  // namespace sherpa_onnx
