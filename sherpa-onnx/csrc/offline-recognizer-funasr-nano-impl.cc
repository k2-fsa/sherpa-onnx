// sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.cc
//
// Copyright (c)  2025  zengyw

#include "sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {
// Build cache_position tensor from attention_mask.
// Creates a [S] int64_t tensor where the first element is the starting position (pos0)
// for writing KV deltas. The remaining elements are consecutive positions [pos0, pos0+1, ..., pos0+S-1].
// For prefill: pos0 = 0, S = context_len
// For decode: pos0 = valid_len - 1, S = 1
static Ort::Value BuildCachePositionFromMask(const Ort::Value &attention_mask,
                                             int32_t seq_len,
                                             OrtAllocator *allocator) {
  auto mask_info = attention_mask.GetTensorTypeAndShapeInfo();
  auto mask_shape = mask_info.GetShape();
  
  // Get the current position from attention_mask length
  // mask_shape is [1, mask_len], where mask_len = past_len + seq_len
  int64_t pos0 = 0;
  if (mask_shape.size() == 2 && mask_shape[1] > 0) {
    // pos0 is the current position in cache (past length = mask_len - seq_len)
    pos0 = static_cast<int64_t>(mask_shape[1]) - seq_len;
  }
  if (pos0 < 0) pos0 = 0;
  
  // Create tensor using allocator
  std::array<int64_t, 1> pos_shape{seq_len};
  Ort::Value cache_position = Ort::Value::CreateTensor<int64_t>(
      allocator, pos_shape.data(), pos_shape.size());
  
  // Fill the tensor with position values
  int64_t *p = cache_position.GetTensorMutableData<int64_t>();
  for (int32_t i = 0; i < seq_len; ++i) {
    p[i] = pos0 + i;
  }
  
  return cache_position;
}

}  // namespace

OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(config),
      config_(config),
      model_(std::make_unique<OfflineFunASRNanoModel>(config.model_config)),
      tokenizer_(std::make_unique<FunASRNanoTokenizer>(
          config.model_config.funasr_nano.tokenizer)),
      rng_(config.model_config.funasr_nano.seed) {
  InitFeatConfig();
}

template <typename Manager>
OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    Manager *mgr, const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(mgr, config),
      config_(config),
      model_(std::make_unique<OfflineFunASRNanoModel>(mgr, config.model_config)),
      tokenizer_(std::make_unique<FunASRNanoTokenizer>(
          mgr, config.model_config.funasr_nano.tokenizer)),
      rng_(config.model_config.funasr_nano.seed) {
  InitFeatConfig();
}

std::unique_ptr<OfflineStream>
OfflineRecognizerFunASRNanoImpl::CreateStream() const {
  return std::make_unique<OfflineStream>(config_.feat_config);
}

// Initialize feature extraction configuration for FunASR-nano.
// Sets normalization, window type, and disables edge snipping and dithering
// to match the model's expected input format.
void OfflineRecognizerFunASRNanoImpl::InitFeatConfig() {
  config_.feat_config.normalize_samples = false;
  config_.feat_config.window_type = "hamming";
  config_.feat_config.snip_edges = false;
  config_.feat_config.dither = 0.0f;
}

// Apply Low Frame Rate (LFR) processing to reduce temporal resolution.
// Concatenates multiple consecutive frames into a single frame.
std::vector<float> OfflineRecognizerFunASRNanoImpl::ApplyLFR(
    const std::vector<float> &in) const {
  int32_t lfr_window_size = model_->LfrWindowSize();
  int32_t lfr_window_shift = model_->LfrWindowShift();
  int32_t in_feat_dim = config_.feat_config.feature_dim;
  int32_t in_num_frames = static_cast<int32_t>(in.size() / in_feat_dim);
  int32_t out_num_frames =
      (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
  if (out_num_frames <= 0) return {};
  int32_t out_feat_dim = in_feat_dim * lfr_window_size;
  std::vector<float> out(out_num_frames * out_feat_dim);
  const float *p_in = in.data();
  float *p_out = out.data();
  for (int32_t i = 0; i != out_num_frames; ++i) {
    std::copy(p_in, p_in + out_feat_dim, p_out);
    p_out += out_feat_dim;
    p_in += lfr_window_shift * in_feat_dim;
  }
  return out;
}

// Build source token IDs with chat template format:
// [system_prompt] [user_prompt] [audio_tokens] [assistant_prompt]
// Returns the token sequence and sets fbank_beg_idx to the start position
// of audio tokens in the sequence.
std::vector<int64_t> OfflineRecognizerFunASRNanoImpl::BuildSourceIds(
    const std::string &system_prompt, const std::string &user_prompt,
    int32_t audio_token_len, int32_t &fbank_beg_idx,
    int32_t &fake_token_len) const {
  const std::string system_text =
      "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
  const std::string user_text = "<|im_start|>user\n" + user_prompt;
  const std::string after_text = "<|im_end|>\n<|im_start|>assistant\n";
  std::vector<int64_t> ids_before =
      tokenizer_->Encode(system_text + user_text);
  std::vector<int64_t> ids_after = tokenizer_->Encode(after_text);
  fbank_beg_idx = static_cast<int32_t>(ids_before.size());
  fake_token_len = audio_token_len;
  int64_t pad_id = tokenizer_->GetPadTokenId();
  if (pad_id < 0) pad_id = tokenizer_->GetEosTokenId();
  std::vector<int64_t> source_ids;
  source_ids.reserve(ids_before.size() + audio_token_len + ids_after.size());
  source_ids.insert(source_ids.end(), ids_before.begin(), ids_before.end());
  // Use pad tokens as placeholders for audio embeddings
  source_ids.insert(source_ids.end(), audio_token_len, pad_id);
  source_ids.insert(source_ids.end(), ids_after.begin(), ids_after.end());
  return source_ids;
}

// Sample token from logits using greedy decoding (argmax).
// Handles both FP16 and FP32 logits, skipping NaN/Inf values.
// Returns token ID 0 as fallback if all logits are invalid.
int64_t OfflineRecognizerFunASRNanoImpl::SampleTokenFromLogitsFp16OrFp32(
    const void *logits, bool is_fp16, int32_t vocab_size) const {
  int32_t best = 0;
  float best_val = -1e30f;
  bool found_valid = false;
  if (is_fp16) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = HalfBitsToFloat(p[i]);
      if (std::isfinite(v) && v > best_val) {
        best_val = v;
        best = i;
        found_valid = true;
      }
    }
  } else {
    const float *p = reinterpret_cast<const float *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      if (std::isfinite(p[i]) && p[i] > best_val) {
        best_val = p[i];
        best = i;
        found_valid = true;
      }
    }
  }
  if (!found_valid) {
    return 0;
  }
  return static_cast<int64_t>(best);
}

// Sample token from logits using temperature and top-p (nucleus) sampling.
// Handles both FP16 and FP32 logits.
// Returns token ID 0 as fallback if all logits are invalid.
// If temperature is very small (<= 1e-6) or invalid, falls back to greedy decoding.
// If top_p >= 1.0, samples from all tokens without sorting (full vocabulary).
int64_t OfflineRecognizerFunASRNanoImpl::SampleTokenWithTemperatureAndTopP(
    const void *logits, bool is_fp16, int32_t vocab_size, float temperature,
    float top_p) const {
  if (temperature <= 1e-6f || !std::isfinite(temperature)) {
    return SampleTokenFromLogitsFp16OrFp32(logits, is_fp16, vocab_size);
  }

  // top_p <= 0: treat as greedy (minimal candidate set)
  if (!std::isfinite(top_p) || top_p <= 0.0f) {
    return SampleTokenFromLogitsFp16OrFp32(logits, is_fp16, vocab_size);
  }
  if (top_p > 1.0f) top_p = 1.0f;

  // Reuse buffers to avoid per-token allocations
  thread_local std::vector<float> probs;
  thread_local std::vector<int32_t> idx;

  probs.resize(vocab_size);
  idx.resize(vocab_size);

  // Scale logits & find max for numerical stability
  float max_logit = -std::numeric_limits<float>::infinity();
  bool found_valid = false;

  if (is_fp16) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = HalfBitsToFloat(p[i]);
      if (std::isfinite(v)) {
        v /= temperature;
        probs[i] = v;
        if (v > max_logit) max_logit = v;
        found_valid = true;
      } else {
        probs[i] = -1e30f;
      }
      idx[i] = i;
    }
  } else {
    const float *p = reinterpret_cast<const float *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = p[i];
      if (std::isfinite(v)) {
        v /= temperature;
        probs[i] = v;
        if (v > max_logit) max_logit = v;
        found_valid = true;
      } else {
        probs[i] = -1e30f;
      }
      idx[i] = i;
    }
  }

  if (!found_valid) return 0;

  // Compute softmax probabilities
  float sum_exp = 0.0f;
  for (int32_t i = 0; i < vocab_size; ++i) {
    float e = std::exp(probs[i] - max_logit);
    probs[i] = e;
    sum_exp += e;
  }
  if (sum_exp <= 0.0f || !std::isfinite(sum_exp)) return 0;
  for (int32_t i = 0; i < vocab_size; ++i) {
    probs[i] /= sum_exp;
  }

  // top_p >= 1: sample from full distribution (no sorting)
  if (top_p >= 1.0f) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(rng_);
    float cumsum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
      cumsum += probs[i];
      if (sample <= cumsum) return static_cast<int64_t>(i);
    }
    return static_cast<int64_t>(vocab_size - 1);
  }

  // Fast path: partial_sort top-K, increase K only if needed
  int32_t k = std::min<int32_t>(256, vocab_size);
  float cum_k = 0.0f;

  while (true) {
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });

    cum_k = 0.0f;
    for (int32_t i = 0; i < k; ++i) cum_k += probs[idx[i]];

    if (cum_k >= top_p || k == vocab_size) break;

    int32_t new_k = std::min(vocab_size, k * 2);
    if (new_k == k) break;
    k = new_k;
  }

  // Find cutoff inside sorted top-K
  float cumsum = 0.0f;
  int32_t cutoff = k;
  for (int32_t i = 0; i < k; ++i) {
    cumsum += probs[idx[i]];
    if (cumsum >= top_p) {
      cutoff = i + 1;
      break;
    }
  }

  float renorm_sum = 0.0f;
  for (int32_t i = 0; i < cutoff; ++i) renorm_sum += probs[idx[i]];
  if (renorm_sum <= 0.0f) return 0;

  std::uniform_real_distribution<float> dist(0.0f, renorm_sum);
  float sample = dist(rng_);
  float cumsum_sample = 0.0f;
  for (int32_t i = 0; i < cutoff; ++i) {
    cumsum_sample += probs[idx[i]];
    if (sample <= cumsum_sample) return static_cast<int64_t>(idx[i]);
  }
  return static_cast<int64_t>(idx[cutoff - 1]);
}

OfflineRecognitionResult OfflineRecognizerFunASRNanoImpl::GenerateText(
    Ort::Value encoder_out, const std::string &system_prompt,
    const std::string &user_prompt) const {
  OfflineRecognitionResult result;
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const auto &funasr_config = config_.model_config.funasr_nano;
  auto enc_shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t audio_token_len = static_cast<int32_t>(enc_shape[1]);
  int32_t hidden_size = static_cast<int32_t>(enc_shape[2]);
  int32_t fbank_beg_idx = 0;
  int32_t fake_token_len = 0;
  std::vector<int64_t> source_ids =
      BuildSourceIds(system_prompt, user_prompt, audio_token_len, fbank_beg_idx, fake_token_len);
  int32_t context_len = static_cast<int32_t>(source_ids.size());

  // Create KV cache buffer [B, max_total_len, kv_h, hd].
  // This stores the accumulated KV cache. Model outputs are deltas that get applied in-place.
  std::vector<std::pair<Ort::Value, Ort::Value>> cache_kv =
      model_->CreateEmptyKVCache(1);
    int32_t max_seq_len = model_->GetMaxTotalLen();
    if (max_seq_len <= 0) {
      SHERPA_ONNX_LOGE("Invalid max_seq_len=%d", max_seq_len);
      result.text = "";
      return result;
    }

  // If context exceeds KV capacity: prioritize truncating audio placeholders
  // (keep prompt scaffold intact).
  if (context_len > max_seq_len) {
    int32_t before_len = fbank_beg_idx;
    int32_t after_len = context_len - before_len - fake_token_len;
    if (after_len < 0) after_len = 0;

    int32_t keep_audio = max_seq_len - before_len - after_len;
    if (keep_audio < 0) {
      SHERPA_ONNX_LOGE(
          "Context_len (%d) too large for KV capacity (%d) and prompts already exceed capacity. "
          "Falling back to keep last %d tokens.",
          context_len, max_seq_len, max_seq_len);
      // Fallback: keep the suffix.
      source_ids.erase(source_ids.begin(), source_ids.end() - max_seq_len);
      // Audio alignment is no longer controllable, skip injecting audio embeddings.
      fbank_beg_idx = -1;
      fake_token_len = 0;
      context_len = static_cast<int32_t>(source_ids.size());
    } else {
      if (keep_audio > audio_token_len) keep_audio = audio_token_len;

      SHERPA_ONNX_LOGE(
          "Context_len (%d) exceeds KV capacity (%d). Truncating audio placeholders: "
          "audio_token_len=%d -> keep_audio=%d (before=%d after=%d).",
          context_len, max_seq_len, audio_token_len, keep_audio, before_len, after_len);

      // Rebuild ids_before/ids_after using slices.
      std::vector<int64_t> ids_before(source_ids.begin(),
                                      source_ids.begin() + before_len);
      std::vector<int64_t> ids_after(source_ids.end() - after_len,
                                     source_ids.end());

      int64_t pad_id = tokenizer_->GetPadTokenId();
      if (pad_id < 0) pad_id = tokenizer_->GetEosTokenId();

      source_ids.clear();
      source_ids.reserve(before_len + keep_audio + after_len);
      source_ids.insert(source_ids.end(), ids_before.begin(), ids_before.end());
      source_ids.insert(source_ids.end(), keep_audio, pad_id);
      source_ids.insert(source_ids.end(), ids_after.begin(), ids_after.end());

      fake_token_len = keep_audio;
      fbank_beg_idx = before_len;
      context_len = static_cast<int32_t>(source_ids.size());
    }
  }

  // Get text embeddings for the prompt tokens
  std::vector<int64_t> input_ids = source_ids;
  std::array<int64_t, 2> ids_shape{1, context_len};
  Ort::Value input_ids_tensor = Ort::Value::CreateTensor(
      memory_info, input_ids.data(), input_ids.size(),
      ids_shape.data(), ids_shape.size());

  Ort::Value text_embeds = model_->ForwardEmbedding(std::move(input_ids_tensor));

  auto te_info = text_embeds.GetTensorTypeAndShapeInfo();
  auto te_shape = te_info.GetShape();

  const auto te_type = te_info.GetElementType();
  const bool te_fp16 = (te_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  // Allocate inputs_embeds only for prefill (context_len * hidden_size).
  // Decode steps will use a separate reusable buffer.
  std::vector<float> inputs_embeds_fp32(
      static_cast<size_t>(context_len) * hidden_size, 0.0f);

  // Copy text embeddings.
  if (te_fp16) {
    const uint16_t *p = text_embeds.GetTensorData<uint16_t>();
    const size_t total = static_cast<size_t>(context_len) * hidden_size;
    for (size_t i = 0; i < total; ++i) {
      inputs_embeds_fp32[i] = HalfBitsToFloat(p[i]);
    }
  } else {
    const float *p = text_embeds.GetTensorData<float>();
    const size_t total = static_cast<size_t>(context_len) * hidden_size;
    std::memcpy(inputs_embeds_fp32.data(), p, total * sizeof(float));
  }

  // Inject audio embeddings into placeholder region (if alignment is still possible).
  auto enc_info2 = encoder_out.GetTensorTypeAndShapeInfo();
  auto enc_et = static_cast<ONNXTensorElementDataType>(enc_info2.GetElementType());
  int32_t copy_len = std::min(fake_token_len, audio_token_len);

  if (copy_len > 0 && fbank_beg_idx >= 0) {
    if (enc_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      const uint16_t *enc = encoder_out.GetTensorData<uint16_t>();
      const size_t hidden_size_u = static_cast<size_t>(hidden_size);
      for (int32_t t = 0; t < copy_len; ++t) {
        const uint16_t *src = enc + static_cast<size_t>(t) * hidden_size_u;
        float *dst = inputs_embeds_fp32.data() +
                     static_cast<size_t>(fbank_beg_idx + t) * hidden_size_u;
        for (size_t d = 0; d < hidden_size_u; ++d) {
          dst[d] = HalfBitsToFloat(src[d]);
        }
      }
    } else if (enc_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const float *enc = encoder_out.GetTensorData<float>();
      const size_t hidden_size_u = static_cast<size_t>(hidden_size);
      for (int32_t t = 0; t < copy_len; ++t) {
        const float *src = enc + static_cast<size_t>(t) * hidden_size_u;
        float *dst = inputs_embeds_fp32.data() +
                     static_cast<size_t>(fbank_beg_idx + t) * hidden_size_u;
        std::memcpy(dst, src, hidden_size_u * sizeof(float));
      }
    } else {
      SHERPA_ONNX_LOGE("encoder_out elem_type=%d not supported", (int)enc_et);
      result.text = "";
      return result;
    }
  }

  // Pre-allocate attention_mask buffer to avoid per-step allocations
  // All values are 1, we'll use a slice of this buffer for each step
  std::vector<int64_t> attention_mask(static_cast<size_t>(max_seq_len), 1);

  // Pre-allocate reusable buffer for decode step embeddings (hidden_size)
  std::vector<float> next_embed_fp32(static_cast<size_t>(hidden_size));

  int32_t valid_len = context_len;

  std::vector<int64_t> generated_ids;
  generated_ids.reserve(funasr_config.max_new_tokens);

  const int64_t eos_id = tokenizer_->GetEosTokenId();
  const int64_t im_end_id = tokenizer_->GetImEndTokenId();
  const int32_t max_new_tokens = funasr_config.max_new_tokens;

  bool is_first_step = true;

  for (int32_t step = 0; step < max_new_tokens; ++step) {
    // valid_len represents the mask_len for the next decode step (= past + current).
    if (valid_len >= max_seq_len) break;

    Ort::Value logits{nullptr};

    if (is_first_step) {
      // Prefill: seq = context_len, mask_len = context_len.
      if (config_.model_config.debug) {
        SHERPA_ONNX_LOGE("GenerateText: starting prefill with context_len=%d, inputs_embeds_fp32.size()=%zu",
           context_len, inputs_embeds_fp32.size());
      }

      std::array<int64_t, 3> embeds_shape{1, context_len, hidden_size};
      Ort::Value inputs_embeds_tensor = Ort::Value::CreateTensor<float>(
          memory_info, inputs_embeds_fp32.data(),
          static_cast<size_t>(context_len) * hidden_size,
          embeds_shape.data(), embeds_shape.size());

      // Use pre-allocated attention_mask buffer (first context_len elements)
      std::array<int64_t, 2> mask_shape{1, context_len};
      Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
          memory_info, attention_mask.data(),
          static_cast<size_t>(context_len),
          mask_shape.data(), mask_shape.size());

      Ort::Value cache_position = BuildCachePositionFromMask(
          attention_mask_tensor, context_len, model_->Allocator());

      auto tmp = model_->ForwardLLM(std::move(inputs_embeds_tensor),
                                    std::move(attention_mask_tensor),
                                    cache_position,
                                    cache_kv);
      logits = std::move(tmp.first);
      auto kv_outputs = std::move(tmp.second);

      // Apply KV deltas to cache buffer in-place.
      // kv_outputs contains deltas that update cache_kv at positions specified by cache_position.
      model_->ApplyKvDeltaInplace(&cache_kv, kv_outputs, cache_position);

    } else {
      // Decode: seq = 1, mask_len = valid_len (= past + 1).
      int64_t last_token_id = generated_ids.back();
      std::vector<int64_t> one_id{last_token_id};
      std::array<int64_t, 2> one_shape{1, 1};
      Ort::Value one_tensor = Ort::Value::CreateTensor(
          memory_info, one_id.data(), one_id.size(),
          one_shape.data(), one_shape.size());

      Ort::Value next_embed = model_->ForwardEmbedding(std::move(one_tensor));
      auto ne_info = next_embed.GetTensorTypeAndShapeInfo();
      bool ne_fp16 = (ne_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

      // Reuse pre-allocated buffer for decode step embedding
      if (ne_fp16) {
        const uint16_t *src = next_embed.GetTensorData<uint16_t>();
        for (size_t d = 0; d < static_cast<size_t>(hidden_size); ++d) {
          next_embed_fp32[d] = HalfBitsToFloat(src[d]);
        }
      } else {
        const float *src = next_embed.GetTensorData<float>();
        std::memcpy(next_embed_fp32.data(), src, static_cast<size_t>(hidden_size) * sizeof(float));
      }

      std::array<int64_t, 3> embeds_shape{1, 1, hidden_size};
      Ort::Value inputs_embeds_tensor = Ort::Value::CreateTensor<float>(
          memory_info, next_embed_fp32.data(),
          static_cast<size_t>(hidden_size),
          embeds_shape.data(), embeds_shape.size());

      // mask_len must equal kv_seq_len (= past + current).
      // Use pre-allocated attention_mask buffer (first valid_len elements)
      std::array<int64_t, 2> mask_shape{1, valid_len};
      Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
          memory_info, attention_mask.data(), static_cast<size_t>(valid_len),
          mask_shape.data(), mask_shape.size());

      Ort::Value cache_position = BuildCachePositionFromMask(
          attention_mask_tensor, 1, model_->Allocator());

      auto tmp = model_->ForwardLLM(std::move(inputs_embeds_tensor),
                                    std::move(attention_mask_tensor),
                                    cache_position,
                                    cache_kv);
      logits = std::move(tmp.first);
      auto kv_outputs = std::move(tmp.second);

      // Apply KV deltas to cache buffer in-place.
      model_->ApplyKvDeltaInplace(&cache_kv, kv_outputs, cache_position);
    }

    auto log_info = logits.GetTensorTypeAndShapeInfo();
    auto log_shape = log_info.GetShape();

    // logits are [B, S, V]. Always pick the last available step.
    if (log_shape.size() < 3) {
      SHERPA_ONNX_LOGE("Unexpected logits rank=%zu", log_shape.size());
      result.text = "";
      return result;
    }

    int32_t time_dim = static_cast<int32_t>(log_shape[1]);
    int32_t vocab_size = static_cast<int32_t>(log_shape[2]);
    if (time_dim <= 0 || vocab_size <= 0) {
      SHERPA_ONNX_LOGE("Invalid logits shape [%lld,%lld,%lld]",
                       (long long)log_shape[0], (long long)log_shape[1],
                       (long long)log_shape[2]);
      result.text = "";
      return result;
    }

    const bool log_fp16 = (log_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

    int32_t last_idx = time_dim - 1;

    const void *base = nullptr;
    if (log_fp16) base = logits.GetTensorData<uint16_t>();
    else base = logits.GetTensorData<float>();

    const size_t offset = static_cast<size_t>(last_idx) * vocab_size;
    const void *last_logits =
        log_fp16
            ? static_cast<const void *>(reinterpret_cast<const uint16_t *>(base) + offset)
            : static_cast<const void *>(reinterpret_cast<const float *>(base) + offset);

    int64_t next_id = SampleTokenWithTemperatureAndTopP(
        last_logits, log_fp16, vocab_size,
        funasr_config.temperature, funasr_config.top_p);

    if (next_id == eos_id || next_id == im_end_id) break;

    generated_ids.push_back(next_id);

    if (is_first_step) is_first_step = false;

    // valid_len represents the kv_seq_len for the next decode step.
    valid_len += 1;
  }

  result.text = tokenizer_->Decode(generated_ids);
  result.text = ApplyInverseTextNormalization(std::move(result.text));
  result.text = ApplyHomophoneReplacer(std::move(result.text));

  if (config_.model_config.debug) {
    SHERPA_ONNX_LOGE("GenerateText: generated %zu tokens: %s",
                     generated_ids.size(), result.text.c_str());
    std::string token_str;
    for (size_t i = 0; i < generated_ids.size() && i < 10; ++i) {
      if (i > 0) token_str += ",";
      token_str += std::to_string(generated_ids[i]);
    }
    SHERPA_ONNX_LOGE("GenerateText: token ids: %s%s",
                     token_str.c_str(), generated_ids.size() > 10 ? "..." : "");
  }

  if (!generated_ids.empty()) {
    result.tokens.reserve(generated_ids.size());
    for (int64_t token_id : generated_ids) {
      std::vector<int64_t> single_token{token_id};
      result.tokens.push_back(tokenizer_->Decode(single_token));
    }

    auto enc_shape2 = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
    int32_t audio_token_len2 = static_cast<int32_t>(enc_shape2[1]);
    int32_t lfr_window_size = model_->LfrWindowSize();
    int32_t lfr_window_shift = model_->LfrWindowShift();

    int32_t original_feature_frames =
        (audio_token_len2 > 0)
            ? ((audio_token_len2 - 1) * lfr_window_shift + lfr_window_size)
            : 0;

    float frame_shift_ms = config_.feat_config.frame_shift_ms;
    float audio_duration =
        (original_feature_frames > 0 && frame_shift_ms > 0)
            ? static_cast<float>(original_feature_frames) * frame_shift_ms / 1000.0f
            : 0.0f;

    result.timestamps.reserve(generated_ids.size());
    if (generated_ids.size() > 1 && audio_duration > 0) {
      float time_per_token = audio_duration / static_cast<float>(generated_ids.size());
      for (size_t i = 0; i < generated_ids.size(); ++i) {
        result.timestamps.push_back(static_cast<float>(i) * time_per_token);
      }
    } else if (generated_ids.size() == 1 && audio_duration > 0) {
      result.timestamps.push_back(audio_duration / 2.0f);
    }
  }

  return result;
}

// Decode multiple audio streams in batch.
// Applies LFR processing, runs encoder, and generates text for each stream.
void OfflineRecognizerFunASRNanoImpl::DecodeStreams(OfflineStream **ss,
                                                    int32_t n) const {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const auto &funasr_config = config_.model_config.funasr_nano;
  for (int32_t i = 0; i != n; ++i) {
    std::vector<float> f = ss[i]->GetFrames();
    f = ApplyLFR(f);
    int32_t num_frames = static_cast<int32_t>(
        f.size() / (config_.feat_config.feature_dim * model_->LfrWindowSize()));
    if (num_frames <= 0) {
      OfflineRecognitionResult r;
      r.text = "";
      ss[i]->SetResult(r);
      continue;
    }

    std::array<int64_t, 3> shape{
        1, num_frames, static_cast<int64_t>(f.size() / num_frames)};

    Ort::Value features = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(f.data()), f.size(),
        shape.data(), shape.size());

    Ort::Value encoder_out =
        model_->ForwardEncoderAdaptor(std::move(features));

    OfflineRecognitionResult r = GenerateText(
        std::move(encoder_out),
        funasr_config.system_prompt,
        funasr_config.user_prompt);

    ss[i]->SetResult(r);
  }
}

#if __ANDROID_API__ >= 9
template OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx
