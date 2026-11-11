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

bool OfflineTtsKittenModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kitten-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--kitten-model: '%s' does not exist", model.c_str());
    return false;
  }

  if (voices.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kitten-voices");
    return false;
  }

  if (!FileExists(voices)) {
    SHERPA_ONNX_LOGE("--kitten-voices: '%s' does not exist", voices.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kitten-tokens");
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--kitten-tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (!data_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "WARNING: --kitten-data-dir is deprecated and ignored in "
        "sherpa-onnx >= v2.0.0. Please use GenerationConfig.phoneme_codepoints "
        "with an external phonemizer (e.g. piper_phonemize) instead. "
        "See python-api-examples/test-offline-tts-kitten-phonemize.py");
  }

  if (!lexicon.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE(
            "lexicon '%s' does not exist. Please re-check --kitten-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (length_scale <= 0) {
    SHERPA_ONNX_LOGE(
        "Please provide a positive length_scale for --kitten-length-scale. "
        "Given: %.3f",
        length_scale);
    return false;
  }

  return true;
}

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
