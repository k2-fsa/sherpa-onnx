// sherpa-onnx/csrc/offline-tts-vits-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-vits-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsVitsModelConfig::Register(ParseOptions *po) {
  po->Register("vits-model", &model, "Path to VITS model");
  po->Register("vits-lexicon", &lexicon, "Path to lexicon.txt for VITS models");
  po->Register("vits-tokens", &tokens, "Path to tokens.txt for VITS models");
  po->Register("vits-data-dir", &data_dir,
               "Path to the directory containing dict for espeak-ng. If it is "
               "given, --vits-lexicon is ignored.");
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
    SHERPA_ONNX_LOGE("--vits-model: %s does not exist", model.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --vits-tokens");
    return false;
  }

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--vits-tokens: %s does not exist", tokens.c_str());
    return false;
  }

  if (!data_dir.empty()) {
    if (!FileExists(data_dir + "/phontab")) {
      SHERPA_ONNX_LOGE("%s/phontab does not exist. Skipping test",
                       data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/phonindex")) {
      SHERPA_ONNX_LOGE("%s/phonindex does not exist. Skipping test",
                       data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/phondata")) {
      SHERPA_ONNX_LOGE("%s/phondata does not exist. Skipping test",
                       data_dir.c_str());
      return false;
    }

    if (!FileExists(data_dir + "/intonations")) {
      SHERPA_ONNX_LOGE("%s/intonations does not exist. Skipping test",
                       data_dir.c_str());
      return false;
    }
  }

  return true;
}

std::string OfflineTtsVitsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsVitsModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "noise_scale=" << noise_scale << ", ";
  os << "noise_scale_w=" << noise_scale_w << ", ";
  os << "length_scale=" << length_scale << ")";

  return os.str();
}

}  // namespace sherpa_onnx
