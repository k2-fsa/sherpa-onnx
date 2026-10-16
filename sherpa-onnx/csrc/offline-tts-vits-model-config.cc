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

bool OfflineTtsVitsModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --vits-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--vits-model: '%s' does not exist", model.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --vits-tokens");
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--vits-tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (!lexicon.empty() && !FileExists(lexicon)) {
    SHERPA_ONNX_LOGE("--vits-lexicon: '%s' does not exist", lexicon.c_str());
    return false;
  }

  if (!data_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "WARNING: --vits-data-dir is deprecated and ignored in "
        "sherpa-onnx >= v2.0.0. Please use GenerationConfig.phoneme_codepoints "
        "with an external phonemizer (e.g. piper_phonemize) instead. "
        "See python-api-examples/test-offline-tts-piper-phonemize.py");
  }

  if (!dict_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "From sherpa-onnx v1.12.15, you don't need to provide dict_dir for "
        "this model. Ignore it");
  }

  return true;
}

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
