#include "sherpa-onnx/csrc/offline-funasr-nano-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineFunASRNanoModelConfig::Register(ParseOptions *po) {
  po->Register("funasr-nano-encoder-adaptor", &encoder_adaptor,
               "Path to encoder_adaptor.onnx for FunASR-nano");

  po->Register("funasr-nano-llm-prefill", &llm_prefill,
               "Path to llm_prefill.onnx for FunASR-nano (KV cache mode)");

  po->Register("funasr-nano-llm-decode", &llm_decode,
               "Path to llm_decode.onnx for FunASR-nano (KV cache mode)");

  po->Register("funasr-nano-embedding", &embedding,
               "Path to embedding.onnx for FunASR-nano (optional)");

  po->Register("funasr-nano-tokenizer", &tokenizer,
               "Path to tokenizer directory (e.g., Qwen3-0.6B) for FunASR-nano");

  po->Register("funasr-nano-system-prompt", &system_prompt,
               "System prompt for FunASR-nano");

  po->Register("funasr-nano-user-prompt", &user_prompt,
               "User prompt template for FunASR-nano");

  po->Register("funasr-nano-max-new-tokens", &max_new_tokens,
               "Maximum number of new tokens to generate for FunASR-nano");

  po->Register("funasr-nano-temperature", &temperature,
               "Sampling temperature for FunASR-nano");

  po->Register("funasr-nano-top-p", &top_p,
               "Top-p (nucleus) sampling threshold for FunASR-nano");

  po->Register("funasr-nano-seed", &seed,
               "Random seed for FunASR-nano");
}

bool OfflineFunASRNanoModelConfig::Validate() const {
  if (encoder_adaptor.empty()) {
    SHERPA_ONNX_LOGE("--funasr-nano-encoder-adaptor is required");
    return false;
  }

  if (!FileExists(encoder_adaptor)) {
    SHERPA_ONNX_LOGE("--funasr-nano-encoder-adaptor: '%s' does not exist",
                     encoder_adaptor.c_str());
    return false;
  }

  // KV cache mode (prefill + decode) is required
  bool use_kv_cache = !llm_prefill.empty() && !llm_decode.empty();

  if (!use_kv_cache) {
    SHERPA_ONNX_LOGE("Both --funasr-nano-llm-prefill and --funasr-nano-llm-decode are required");
    return false;
  }

  if (use_kv_cache) {
    if (!FileExists(llm_prefill)) {
      SHERPA_ONNX_LOGE("--funasr-nano-llm-prefill: '%s' does not exist", llm_prefill.c_str());
      return false;
    }
    if (!FileExists(llm_decode)) {
      SHERPA_ONNX_LOGE("--funasr-nano-llm-decode: '%s' does not exist", llm_decode.c_str());
      return false;
    }
  }

  if (tokenizer.empty()) {
    SHERPA_ONNX_LOGE("--funasr-nano-tokenizer is required");
    return false;
  }

  if (!FileExists(tokenizer)) {
    SHERPA_ONNX_LOGE("--funasr-nano-tokenizer: '%s' does not exist",
                     tokenizer.c_str());
    return false;
  }

  if (!embedding.empty() && !FileExists(embedding)) {
    SHERPA_ONNX_LOGE("--funasr-nano-embedding: '%s' does not exist",
                     embedding.c_str());
    return false;
  }

  if (max_new_tokens <= 0) {
    SHERPA_ONNX_LOGE("--funasr-nano-max-new-tokens should be > 0. Given: %d",
                     max_new_tokens);
    return false;
  }

  if (temperature < 0.0f) {
    SHERPA_ONNX_LOGE("--funasr-nano-temperature should be >= 0.0. Given: %f",
                     temperature);
    return false;
  }

  if (top_p < 0.0f || top_p > 1.0f) {
    SHERPA_ONNX_LOGE("--funasr-nano-top-p should be in [0.0, 1.0]. Given: %f",
                     top_p);
    return false;
  }

  return true;
}

std::string OfflineFunASRNanoModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineFunASRNanoModelConfig(";
  os << "encoder_adaptor=\"" << encoder_adaptor << "\", ";
  os << "llm_prefill=\"" << llm_prefill << "\", ";
  os << "llm_decode=\"" << llm_decode << "\", ";
  os << "embedding=\"" << embedding << "\", ";
  os << "tokenizer=\"" << tokenizer << "\", ";
  os << "system_prompt=\"" << system_prompt << "\", ";
  os << "user_prompt=\"" << user_prompt << "\", ";
  os << "max_new_tokens=" << max_new_tokens << ", ";
  os << "temperature=" << temperature << ", ";
  os << "top_p=" << top_p << ", ";
  os << "seed=" << seed << ")";

  return os.str();
}

}  // namespace sherpa_onnx

