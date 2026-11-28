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
  po->Register("pocket-text-conditioner", &text_conditioner,
               "Path to PocketTTS text conditioner model");
  po->Register("pocket-vocab-json", &vocab_json,
               "Path to PocketTTS vocab.json");
  po->Register("pocket-token-scores-json", &token_scores_json,
               "Path to PocketTTS token_scores.json");
  po->Register("pocket-voice-embedding-cache-capacity",
               &voice_embedding_cache_capacity,
               "Capacity of the voice embedding cache (number of items). "
               "Default: 50. 0 disables caching.");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOfflineTtsPocketModelConfig(
    const OfflineTtsPocketModelConfig &config, const Exists &exists) {
  if (config.lm_flow.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-lm-flow");
    return false;
  }

  if (!exists(config.lm_flow)) {
    SHERPA_ONNX_LOGE("--pocket-lm-flow '%s' does not exist",
                     config.lm_flow.c_str());
    return false;
  }

  if (config.lm_main.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-lm-main");
    return false;
  }

  if (!exists(config.lm_main)) {
    SHERPA_ONNX_LOGE("--pocket-lm-main '%s' does not exist",
                     config.lm_main.c_str());
    return false;
  }

  if (config.encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-encoder");
    return false;
  }

  if (!exists(config.encoder)) {
    SHERPA_ONNX_LOGE("--pocket-encoder '%s' does not exist",
                     config.encoder.c_str());
    return false;
  }

  if (config.decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-decoder");
    return false;
  }

  if (!exists(config.decoder)) {
    SHERPA_ONNX_LOGE("--pocket-decoder '%s' does not exist",
                     config.decoder.c_str());
    return false;
  }

  if (config.text_conditioner.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-text-conditioner");
    return false;
  }

  if (!exists(config.text_conditioner)) {
    SHERPA_ONNX_LOGE("--pocket-text-conditioner '%s' does not exist",
                     config.text_conditioner.c_str());
    return false;
  }

  if (config.vocab_json.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-vocab-json");
    return false;
  }

  if (!exists(config.vocab_json)) {
    SHERPA_ONNX_LOGE("--pocket-vocab-json '%s' does not exist",
                     config.vocab_json.c_str());
    return false;
  }

  if (config.token_scores_json.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-token-scores-json");
    return false;
  }

  if (!exists(config.token_scores_json)) {
    SHERPA_ONNX_LOGE("--pocket-token-scores-json '%s' does not exist",
                     config.token_scores_json.c_str());
    return false;
  }

  if (config.voice_embedding_cache_capacity < 0) {
    SHERPA_ONNX_LOGE(
        "voice_embedding_cache_capacity must be non-negative. Given: %d",
        config.voice_embedding_cache_capacity);
    return false;
  }

  return true;
}

bool OfflineTtsPocketModelConfig::Validate() const {
  return ValidateOfflineTtsPocketModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineTtsPocketModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineTtsPocketModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OfflineTtsPocketModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineTtsPocketModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineTtsPocketModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsPocketModelConfig(";
  os << "lm_flow=\"" << lm_flow << "\", ";
  os << "lm_main=\"" << lm_main << "\", ";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "text_conditioner=\"" << text_conditioner << "\", ";
  os << "vocab_json=\"" << vocab_json << "\", ";
  os << "token_scores_json=\"" << token_scores_json << "\", ";
  os << "voice_embedding_cache_capacity=" << voice_embedding_cache_capacity
     << ")";

  return os.str();
}

}  // namespace sherpa_onnx
