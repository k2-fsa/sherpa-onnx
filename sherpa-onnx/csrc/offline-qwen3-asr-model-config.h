// sherpa-onnx/csrc/offline-qwen3-asr-model-config.h
//
// Copyright (c)  2026  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_QWEN3_ASR_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_QWEN3_ASR_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineQwen3ASRModelConfig {
  // Path to conv_frontend.onnx
  std::string conv_frontend;

  // Path to encoder.onnx
  std::string encoder;

  // Path to decoder.onnx (KV cache model)
  std::string decoder;

  // Path to tokenizer directory (e.g., Qwen3-ASR-0.6B)
  std::string tokenizer;

  // Optional comma-separated hotwords (UTF-8, ASCII ','), e.g. "foo,bar,baz".
  std::string hotwords;

  // Maximum total sequence length (from model metadata or config).
  // 1024 holds one full 30-second clip (~390 audio tokens at 13 tokens/s
  // plus the prompt scaffold) plus generation headroom. The old default
  // 512 left a 30-second clip only ~107 of the default 128 generation
  // tokens (generation then stops silently mid-word) and truncated audio
  // placeholders for clips longer than ~38s. Note the cost: on
  // dynamic-dim exports (e.g. the official 0.6B int8) the KV cache is
  // allocated at this size, roughly 110 MiB per 512 positions (float32,
  // 28 layers), so 1024 doubles the per-recognizer transient versus 512.
  int32_t max_total_len = 1024;

  // Maximum number of new tokens to generate
  int32_t max_new_tokens = 128;

  // Sampling temperature
  float temperature = 1e-6f;

  // Top-p (nucleus) sampling threshold
  float top_p = 0.8f;

  // Random seed for reproducibility
  int32_t seed = 42;

  OfflineQwen3ASRModelConfig() = default;

  OfflineQwen3ASRModelConfig(const std::string &conv_frontend,
                             const std::string &encoder,
                             const std::string &decoder,
                             const std::string &tokenizer,
                             int32_t max_total_len, int32_t max_new_tokens,
                             float temperature, float top_p, int32_t seed,
                             const std::string &hotwords = "")
      : conv_frontend(conv_frontend),
        encoder(encoder),
        decoder(decoder),
        tokenizer(tokenizer),
        hotwords(hotwords),
        max_total_len(max_total_len),
        max_new_tokens(max_new_tokens),
        temperature(temperature),
        top_p(top_p),
        seed(seed) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_QWEN3_ASR_MODEL_CONFIG_H_
