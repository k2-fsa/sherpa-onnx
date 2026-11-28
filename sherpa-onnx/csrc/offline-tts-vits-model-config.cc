// sherpa-onnx/csrc/offline-tts-vits-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-vits-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsVitsModelConfig::Register(ParseOptions *po) {
  po->Register("vits-model", &model, "Path to VITS model");
  po->Register("vits-lexicon", &lexicon, "Path to lexicon.txt for VITS models");
  po->Register("vits-tokens", &tokens, "Path to tokens.txt for VITS models");
  po->Register("vits-data-dir", &data_dir,
               "Deprecated. Ignored since v2.0.0: espeak-ng is no longer used. "
               "Use --vits-lexicon or "
               "GenerationConfig.phoneme_codepoints instead.");
  po->Register("vits-dict-dir", &dict_dir,
               "Not used. You don't need to provide a value for it");
  po->Register("vits-noise-scale", &noise_scale, "noise_scale for VITS models");
  po->Register("vits-noise-scale-w", &noise_scale_w,
               "noise_scale_w for VITS models");
  po->Register("vits-length-scale", &length_scale,
               "Speech speed. Larger->Slower; Smaller->faster.");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOfflineTtsVitsModelConfig(
    const OfflineTtsVitsModelConfig &config, const Exists &exists) {
  if (config.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --vits-model");
    return false;
  }

  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("--vits-model: '%s' does not exist", config.model.c_str());
    return false;
  }

  if (config.tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --vits-tokens");
    return false;
  }

  if (!exists(config.tokens)) {
    SHERPA_ONNX_LOGE("--vits-tokens: '%s' does not exist",
                     config.tokens.c_str());
    return false;
  }

  if (!config.lexicon.empty() && !exists(config.lexicon)) {
    SHERPA_ONNX_LOGE("--vits-lexicon: '%s' does not exist",
                     config.lexicon.c_str());
    return false;
  }

  if (!config.data_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "WARNING: --vits-data-dir is deprecated and ignored in "
        "sherpa-onnx >= v2.0.0. Please use GenerationConfig.phoneme_codepoints "
        "with an external phonemizer (e.g. piper_phonemize) instead. "
        "See python-api-examples/test-offline-tts-piper-phonemize.py");
  }

  if (!config.dict_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "From sherpa-onnx v1.12.15, you don't need to provide dict_dir for "
        "this model. Ignore it");
  }

  return true;
}

bool OfflineTtsVitsModelConfig::Validate() const {
  return ValidateOfflineTtsVitsModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineTtsVitsModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineTtsVitsModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OfflineTtsVitsModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineTtsVitsModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineTtsVitsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsVitsModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "noise_scale=" << noise_scale << ", ";
  os << "noise_scale_w=" << noise_scale_w << ", ";
  os << "length_scale=" << length_scale << ")";

  return os.str();
}

}  // namespace sherpa_onnx
