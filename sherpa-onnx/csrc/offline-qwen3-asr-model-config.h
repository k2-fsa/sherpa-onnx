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

  // Maximum total sequence length (from model metadata or config)
  int32_t max_total_len = 512;

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
                             float temperature, float top_p, int32_t seed)
      : conv_frontend(conv_frontend),
        encoder(encoder),
        decoder(decoder),
        tokenizer(tokenizer),
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
