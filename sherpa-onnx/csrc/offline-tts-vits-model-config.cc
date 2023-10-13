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

  if (lexicon.empty()) {
    SHERPA_ONNX_LOGE("Please provide --vits-lexicon");
    return false;
  }

  if (!FileExists(lexicon)) {
    SHERPA_ONNX_LOGE("--vits-lexicon: %s does not exist", lexicon.c_str());
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

  return true;
}

std::string OfflineTtsVitsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsVitsModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "tokens=\"" << tokens << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
