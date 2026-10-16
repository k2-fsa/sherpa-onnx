// sherpa-onnx/csrc/offline-tts-kokoro-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-kokoro-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineTtsKokoroModelConfig::Register(ParseOptions *po) {
  po->Register("kokoro-model", &model, "Path to Kokoro model");
  po->Register("kokoro-voices", &voices,
               "Path to voices.bin for Kokoro models");
  po->Register("kokoro-tokens", &tokens,
               "Path to tokens.txt for Kokoro models");
  po->Register("kokoro-lang", &lang,
               "Used only by kokoro >= 1.0. Example values: "
               "en (English), "
               "es (Spanish), fr (French), hi (hindi), it (Italian), "
               "pt-br (Brazilian Portuguese)."
               "You can leave it empty, in which case you need to provide "
               "--kokoro-lexicon.");
  po->Register(
      "kokoro-lexicon", &lexicon,
      "Path to lexicon.txt for Kokoro models. Used only for Kokoro >= v1.0"
      "You can pass multiple files, separated by ','. Example: "
      "./lexicon-us-en.txt,./lexicon-zh.txt");
  po->Register("kokoro-data-dir", &data_dir,
               "Deprecated. Ignored since v2.0.0: espeak-ng is no longer used. "
               "Use --kokoro-lexicon or "
               "GenerationConfig.phoneme_codepoints instead.");
  po->Register("kokoro-dict-dir", &dict_dir,
               "Not used. You don't need to provide a value for it");
  po->Register("kokoro-length-scale", &length_scale,
               "Speech speed. Larger->Slower; Smaller->faster.");
}

bool OfflineTtsKokoroModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kokoro-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--kokoro-model: '%s' does not exist", model.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kokoro-tokens");
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--kokoro-tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (voices.empty()) {
    SHERPA_ONNX_LOGE("Please provide --kokoro-voices");
    return false;
  }

  if (!FileExists(voices)) {
    SHERPA_ONNX_LOGE("--kokoro-voices: '%s' does not exist", voices.c_str());
    return false;
  }

  if (!lexicon.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE(
            "lexicon '%s' does not exist. Please re-check --kokoro-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (!data_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "WARNING: --kokoro-data-dir is deprecated and ignored in "
        "sherpa-onnx >= v2.0.0. Please use --kokoro-lexicon or "
        "GenerationConfig.phoneme_codepoints with an external "
        "phonemizer (e.g. piper_phonemize) instead.");
  }

  if (!lang.empty()) {
    SHERPA_ONNX_LOGE(
        "WARNING: --kokoro-lang is deprecated and ignored in "
        "sherpa-onnx >= v2.0.0. espeak-ng is no longer used. "
        "Language is determined by the lexicon files you provide.");
  }

  if (!dict_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "From sherpa-onnx v1.12.15, you don't need to provide dict_dir or "
        "dictDir for this model. Ignore this value.");
  }

  return true;
}

std::string OfflineTtsKokoroModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsKokoroModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "voices=\"" << voices << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "length_scale=" << length_scale << ", ";
  os << "lang=\"" << lang << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
