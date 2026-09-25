// sherpa-onnx/csrc/offline-qwen3-asr-model-config.cc
//
// Copyright (c)  2026  zengyw

#include "sherpa-onnx/csrc/offline-qwen3-asr-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineQwen3ASRModelConfig::Register(ParseOptions *po) {
  po->Register("qwen3-asr-conv-frontend", &conv_frontend,
               "Path to conv_frontend.onnx for Qwen3-ASR");

  po->Register("qwen3-asr-encoder", &encoder,
               "Path to encoder.onnx for Qwen3-ASR");

  po->Register("qwen3-asr-decoder", &decoder,
               "Path to decoder.onnx for Qwen3-ASR (KV cache mode)");

  po->Register(
      "qwen3-asr-tokenizer", &tokenizer,
      "Path to tokenizer directory (e.g., Qwen3-ASR-0.6B) for Qwen3-ASR");

  po->Register("qwen3-asr-hotwords", &hotwords,
               "Optional comma-separated hotwords (UTF-8, ASCII ','), e.g. "
               "\"foo,bar,baz\".");

  po->Register("qwen3-asr-max-total-len", &max_total_len,
               "Maximum total sequence length for Qwen3-ASR");

  po->Register("qwen3-asr-max-new-tokens", &max_new_tokens,
               "Maximum number of new tokens to generate for Qwen3-ASR");

  po->Register("qwen3-asr-temperature", &temperature,
               "Sampling temperature for Qwen3-ASR");

  po->Register("qwen3-asr-top-p", &top_p,
               "Top-p (nucleus) sampling threshold for Qwen3-ASR");

  po->Register("qwen3-asr-seed", &seed, "Random seed for Qwen3-ASR");

  po->Register("qwen3-asr-forced-aligner-conv-frontend",
               &forced_aligner_conv_frontend,
               "Optional. Path to conv_frontend.onnx for Qwen3-ForcedAligner. "
               "Used together with --qwen3-asr-forced-aligner-encoder and "
               "--qwen3-asr-forced-aligner-decoder to compute word-level "
               "timestamps");

  po->Register("qwen3-asr-forced-aligner-encoder", &forced_aligner_encoder,
               "Optional. Path to encoder.onnx for Qwen3-ForcedAligner");

  po->Register("qwen3-asr-forced-aligner-decoder", &forced_aligner_decoder,
               "Optional. Path to decoder.onnx for Qwen3-ForcedAligner "
               "(single forward pass, no KV cache)");

  po->Register("qwen3-asr-forced-aligner-tokenizer", &forced_aligner_tokenizer,
               "Optional. Path to tokenizer directory of "
               "Qwen3-ForcedAligner (contains the <timestamp> token)");
}

bool OfflineQwen3ASRModelConfig::Validate() const {
  if (conv_frontend.empty()) {
    SHERPA_ONNX_LOGE("--qwen3-asr-conv-frontend is required");
    return false;
  }

  if (!FileExists(conv_frontend)) {
    SHERPA_ONNX_LOGE("--qwen3-asr-conv-frontend: '%s' does not exist",
                     conv_frontend.c_str());
    return false;
  }

  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("--qwen3-asr-encoder is required");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("--qwen3-asr-encoder: '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("--qwen3-asr-decoder is required");
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("--qwen3-asr-decoder: '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  if (tokenizer.empty()) {
    SHERPA_ONNX_LOGE("--qwen3-asr-tokenizer is required");
    return false;
  }

  if (!FileExists(tokenizer + "/vocab.json")) {
    SHERPA_ONNX_LOGE(
        "'%s/vocab.json' does not exist. Please check --qwen3-asr-tokenizer",
        tokenizer.c_str());
    return false;
  }

  if (!FileExists(tokenizer + "/merges.txt")) {
    SHERPA_ONNX_LOGE(
        "'%s/merges.txt' does not exist. Please check --qwen3-asr-tokenizer",
        tokenizer.c_str());
    return false;
  }

  if (!FileExists(tokenizer + "/tokenizer_config.json")) {
    SHERPA_ONNX_LOGE(
        "'%s/tokenizer_config.json' does not exist. Please check "
        "--qwen3-asr-tokenizer",
        tokenizer.c_str());
    return false;
  }

  if (max_total_len <= 0) {
    SHERPA_ONNX_LOGE("--qwen3-asr-max-total-len should be > 0. Given: %d",
                     max_total_len);
    return false;
  }

  if (max_new_tokens <= 0) {
    SHERPA_ONNX_LOGE("--qwen3-asr-max-new-tokens should be > 0. Given: %d",
                     max_new_tokens);
    return false;
  }

  if (temperature < 0.0f) {
    SHERPA_ONNX_LOGE("--qwen3-asr-temperature should be >= 0.0. Given: %f",
                     temperature);
    return false;
  }

  if (top_p < 0.0f || top_p > 1.0f) {
    SHERPA_ONNX_LOGE("--qwen3-asr-top-p should be in [0.0, 1.0]. Given: %f",
                     top_p);
    return false;
  }

  const bool has_any_aligner = !forced_aligner_conv_frontend.empty() ||
                               !forced_aligner_encoder.empty() ||
                               !forced_aligner_decoder.empty() ||
                               !forced_aligner_tokenizer.empty();
  if (has_any_aligner) {
    if (forced_aligner_conv_frontend.empty() ||
        forced_aligner_encoder.empty() || forced_aligner_decoder.empty() ||
        forced_aligner_tokenizer.empty()) {
      SHERPA_ONNX_LOGE(
          "Please provide all of --qwen3-asr-forced-aligner-conv-frontend, "
          "--qwen3-asr-forced-aligner-encoder, "
          "--qwen3-asr-forced-aligner-decoder and "
          "--qwen3-asr-forced-aligner-tokenizer, or none of them");
      return false;
    }

    if (!FileExists(forced_aligner_conv_frontend)) {
      SHERPA_ONNX_LOGE(
          "--qwen3-asr-forced-aligner-conv-frontend: '%s' does not exist",
          forced_aligner_conv_frontend.c_str());
      return false;
    }

    if (!FileExists(forced_aligner_encoder)) {
      SHERPA_ONNX_LOGE(
          "--qwen3-asr-forced-aligner-encoder: '%s' does not exist",
          forced_aligner_encoder.c_str());
      return false;
    }

    if (!FileExists(forced_aligner_decoder)) {
      SHERPA_ONNX_LOGE(
          "--qwen3-asr-forced-aligner-decoder: '%s' does not exist",
          forced_aligner_decoder.c_str());
      return false;
    }

    if (!FileExists(forced_aligner_tokenizer + "/vocab.json") ||
        !FileExists(forced_aligner_tokenizer + "/merges.txt") ||
        !FileExists(forced_aligner_tokenizer + "/tokenizer_config.json")) {
      SHERPA_ONNX_LOGE(
          "'%s' is not a valid tokenizer directory (need vocab.json, "
          "merges.txt and tokenizer_config.json). Please check "
          "--qwen3-asr-forced-aligner-tokenizer",
          forced_aligner_tokenizer.c_str());
      return false;
    }
  }

  return true;
}

std::string OfflineQwen3ASRModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineQwen3ASRModelConfig(";
  os << "conv_frontend=\"" << conv_frontend << "\", ";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "tokenizer=\"" << tokenizer << "\", ";
  os << "hotwords=\"" << hotwords << "\", ";
  os << "max_total_len=" << max_total_len << ", ";
  os << "max_new_tokens=" << max_new_tokens << ", ";
  os << "temperature=" << temperature << ", ";
  os << "top_p=" << top_p << ", ";
  os << "seed=" << seed << ", ";
  os << "forced_aligner_conv_frontend=\"" << forced_aligner_conv_frontend
     << "\", ";
  os << "forced_aligner_encoder=\"" << forced_aligner_encoder << "\", ";
  os << "forced_aligner_decoder=\"" << forced_aligner_decoder << "\", ";
  os << "forced_aligner_tokenizer=\"" << forced_aligner_tokenizer << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
