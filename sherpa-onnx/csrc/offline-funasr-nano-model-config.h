// sherpa-onnx/csrc/offline-funasr-nano-model-config.h
//
// Copyright (c)  2025  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineFunASRNanoModelConfig {
  // Path to encoder_adaptor.onnx
  std::string encoder_adaptor;

  // Path to llm_prefill.onnx (KV cache prefill)
  std::string llm_prefill;

  // Path to llm_decode.onnx (KV cache decode)
  std::string llm_decode;

  // Path to embedding.onnx
  std::string embedding;

  // Path to tokenizer directory (e.g., Qwen3-0.6B)
  std::string tokenizer;

  // System prompt
  std::string system_prompt = "You are a helpful assistant.";

  // User prompt template (will be filled with audio tokens)
  std::string user_prompt = "语音转写：";

  // Maximum number of new tokens to generate
  int32_t max_new_tokens = 512;

  // Sampling temperature
  float temperature = 0.3f;

  // Top-p (nucleus) sampling threshold
  float top_p = 0.8f;

  // Random seed for reproducibility
  int32_t seed = 42;

  OfflineFunASRNanoModelConfig() = default;

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_CONFIG_H_

