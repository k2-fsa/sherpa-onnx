// sherpa-onnx/csrc/offline-tts-matcha-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-matcha-model-config.h"

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
               "Path to the directory containing dict for espeak-ng. If it is "
               "given, --matcha-lexicon is ignored.");
  po->Register("matcha-dict-dir", &dict_dir,
               "Not used. You don't need to provide a value for it");
  po->Register("matcha-noise-scale", &noise_scale,
               "noise_scale for Matcha models");
  po->Register("matcha-length-scale", &length_scale,
               "Speech speed. Larger->Slower; Smaller->faster.");
}

bool OfflineTtsMatchaModelConfig::Validate() const {
  if (acoustic_model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --matcha-acoustic-model");
    return false;
  }

  if (!FileExists(acoustic_model)) {
    SHERPA_ONNX_LOGE("--matcha-acoustic-model: '%s' does not exist",
                     acoustic_model.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --matcha-tokens");
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--matcha-tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (!data_dir.empty()) {
    if (!FileExists(data_dir + "/phontab")) {
      SHERPA_ONNX_LOGE(
          "'%s/phontab' does not exist. Please check --matcha-data-dir",
          data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/phonindex")) {
      SHERPA_ONNX_LOGE(
          "'%s/phonindex' does not exist. Please check --matcha-data-dir",
          data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/phondata")) {
      SHERPA_ONNX_LOGE(
          "'%s/phondata' does not exist. Please check --matcha-data-dir",
          data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/intonations")) {
      SHERPA_ONNX_LOGE(
          "'%s/intonations' does not exist. Please check --matcha-data-dir",
          data_dir.c_str());
      return false;
    }
  }

  if (!lexicon.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE(
            "lexicon '%s' does not exist. Please re-check --matcha-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (!dict_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "From sherpa-onnx v1.12.15, you don't need to provide dict_dir for "
        "this model. Ignore it");
  }

  return true;
}

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
