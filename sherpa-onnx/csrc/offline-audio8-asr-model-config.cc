// sherpa-onnx/csrc/offline-audio8-asr-model-config.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/csrc/offline-audio8-asr-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineAudio8AsrModelConfig::Register(ParseOptions *po) {
  po->Register("audio8-asr-encoder", &encoder,
               "Path to encoder.onnx for Audio8-ASR");

  po->Register("audio8-asr-decoder", &decoder,
               "Path to decoder.onnx for Audio8-ASR (KV cache mode)");

  po->Register("audio8-asr-tokenizer", &tokenizer,
               "Path to the tokenizer directory for Audio8-ASR "
               "(tokenizer.json, vocab.json, merges.txt, "
               "tokenizer_config.json)");

  po->Register("audio8-asr-max-total-len", &max_total_len,
               "Number of KV cache slots in the Audio8-ASR decoder graph");

  po->Register("audio8-asr-max-new-tokens", &max_new_tokens,
               "Maximum number of new tokens to generate for Audio8-ASR");
}

bool OfflineAudio8AsrModelConfig::Validate() const {
  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("--audio8-asr-encoder is required");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("--audio8-asr-encoder: '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("--audio8-asr-decoder is required");
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("--audio8-asr-decoder: '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  if (tokenizer.empty()) {
    SHERPA_ONNX_LOGE("--audio8-asr-tokenizer is required");
    return false;
  }

  if (!FileExists(tokenizer + "/vocab.json")) {
    SHERPA_ONNX_LOGE(
        "'%s/vocab.json' does not exist. Please check --audio8-asr-tokenizer",
        tokenizer.c_str());
    return false;
  }

  if (!FileExists(tokenizer + "/merges.txt")) {
    SHERPA_ONNX_LOGE(
        "'%s/merges.txt' does not exist. Please check --audio8-asr-tokenizer",
        tokenizer.c_str());
    return false;
  }

  if (!FileExists(tokenizer + "/tokenizer_config.json")) {
    SHERPA_ONNX_LOGE(
        "'%s/tokenizer_config.json' does not exist. Without it the added "
        "tokens (<|audio|>, <|begin_of_audio|>, ...) are silently lost. "
        "Please check --audio8-asr-tokenizer",
        tokenizer.c_str());
    return false;
  }

  if (max_total_len <= 0) {
    SHERPA_ONNX_LOGE("--audio8-asr-max-total-len should be > 0. Given: %d",
                     max_total_len);
    return false;
  }

  if (max_new_tokens <= 0) {
    SHERPA_ONNX_LOGE("--audio8-asr-max-new-tokens should be > 0. Given: %d",
                     max_new_tokens);
    return false;
  }

  return true;
}

std::string OfflineAudio8AsrModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineAudio8AsrModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "tokenizer=\"" << tokenizer << "\", ";
  os << "max_total_len=" << max_total_len << ", ";
  os << "max_new_tokens=" << max_new_tokens << ")";

  return os.str();
}

}  // namespace sherpa_onnx
