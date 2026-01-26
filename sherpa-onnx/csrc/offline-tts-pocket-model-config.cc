// sherpa-onnx/csrc/offline-tts-pocket-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-pocket-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsPocketModelConfig::Register(ParseOptions *po) {
  po->Register("pocket-lm-flow", &lm_flow, "Path to PocketTTS lm flow model");
  po->Register("pocket-lm-main", &lm_main, "Path to PocketTTS lm main model");
  po->Register("pocket-encoder", &encoder, "Path to PocketTTS encoder model");
  po->Register("pocket-decoder", &decoder, "Path to PocketTTS decoder model");
  po->Register("pocket-text-conditioner", &decoder,
               "Path to PocketTTS text conditioner model");
  po->Register("pocket-vocab-json", &vocab_json,
               "Path to PocketTTS vocab.json");
  po->Register("pocket-token-scores-json", &token_scores_json,
               "Path to PocketTTS token_scores.json");
}

bool OfflineTtsPocketModelConfig::Validate() const {
  if (lm_flow.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-lm-flow");
    return false;
  }

  if (!FileExists(lm_flow)) {
    SHERPA_ONNX_LOGE("--pocket-lm-flow '%s' does not exist", lm_flow.c_str());
    return false;
  }

  if (lm_main.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-lm-main");
    return false;
  }

  if (!FileExists(lm_main)) {
    SHERPA_ONNX_LOGE("--pocket-lm-main '%s' does not exist", lm_main.c_str());
    return false;
  }

  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-encoder");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("--pocket-encoder '%s' does not exist", encoder.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-decoder");
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("--pocket-decoder '%s' does not exist", decoder.c_str());
    return false;
  }

  if (text_conditioner.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-text-conditioner");
    return false;
  }

  if (!FileExists(text_conditioner)) {
    SHERPA_ONNX_LOGE("--pocket-text-conditioner '%s' does not exist",
                     text_conditioner.c_str());
    return false;
  }

  if (vocab_json.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-vocab-json");
    return false;
  }

  if (!FileExists(vocab_json)) {
    SHERPA_ONNX_LOGE("--pocket-vocab-json '%s' does not exist",
                     vocab_json.c_str());
    return false;
  }

  if (token_scores_json.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-token-scores-json");
    return false;
  }

  if (!FileExists(token_scores_json)) {
    SHERPA_ONNX_LOGE("--pocket-token-scores-json '%s' does not exist",
                     token_scores_json.c_str());
    return false;
  }

  return true;
}

std::string OfflineTtsPocketModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsPocketModelConfig(";
  os << "lm_flow=\"" << lm_flow << "\", ";
  os << "lm_main=\"" << lm_main << "\", ";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "text_conditioner=\"" << text_conditioner << "\", ";
  os << "vocab_json=\"" << vocab_json << "\", ";
  os << "token_scores_json=\"" << token_scores_json << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
