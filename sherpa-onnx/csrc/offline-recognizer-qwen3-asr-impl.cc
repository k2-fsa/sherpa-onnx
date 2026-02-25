// sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.cc
//
// Copyright (c)  2026 zengyw

#include "sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h"

#include <algorithm>
#include <cctype>
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
#include "kaldi-native-fbank/csrc/online-feature.h"

namespace sherpa_onnx {

namespace {

// Compute the number of audio tokens from the number of feature frames
// Aligned with Python: _feat_to_audio_tokens_len_np
int32_t FeatToAudioTokensLen(int32_t feat_len, int32_t chunk_size) {
  // Python's _conv_out_len_3x_stride2:
  // x = (n + 1) // 2
  // x = (x + 1) // 2
  // return (x + 1) // 2
  auto conv_out_len_3x_stride2 = [](int32_t n) -> int32_t {
    int32_t x = (n + 1) / 2;
    x = (x + 1) / 2;
    return (x + 1) / 2;
  };

  // Python's _aftercnn:
  // x = (x - 1) // 2 + 1
  // x = (x - 1) // 2 + 1
  // return (x - 1) // 2 + 1
  auto aftercnn = [](int32_t x) -> int32_t {
    x = (x - 1) / 2 + 1;
    x = (x - 1) / 2 + 1;
    return (x - 1) / 2 + 1;
  };

  int32_t cs = chunk_size;
  int32_t full = feat_len / cs;
  int32_t rem = feat_len % cs;
  int32_t tn = conv_out_len_3x_stride2(cs);
  int32_t out = full * tn + aftercnn(rem);
  return std::max(out, 0);
}

// Trim audio features by removing padding frames with low energy
// This matches Python's _trim_audio_features function
Ort::Value TrimAudioFeatures(Ort::Value audio_features,
                             OrtAllocator *allocator) {
  auto shape = audio_features.GetTensorTypeAndShapeInfo().GetShape();
  int32_t A = static_cast<int32_t>(shape[1]);
  int32_t H = static_cast<int32_t>(shape[2]);

  const float *data = audio_features.GetTensorData<float>();

  // Find the last valid frame based on energy
  int32_t A_valid = A;
  const float eps = 1e-6f;
  for (int32_t a = A - 1; a >= 0; --a) {
    float max_energy = 0.0f;
    for (int64_t h = 0; h < H; ++h) {
      float abs_val = std::abs(data[a * H + h]);
      if (abs_val > max_energy) max_energy = abs_val;
    }
    if (max_energy > eps) {
      A_valid = a + 1;
      break;
    }
  }

  if (A_valid == A) return audio_features;

  // Create trimmed tensor
  std::array<int64_t, 3> new_shape{1, static_cast<int64_t>(A_valid), H};
  Ort::Value trimmed = Ort::Value::CreateTensor<float>(
      allocator, new_shape.data(), new_shape.size());

  float *dst = trimmed.GetTensorMutableData<float>();
  const float *src = data;
  std::memcpy(dst, src, static_cast<size_t>(A_valid) * static_cast<size_t>(H) *
                            sizeof(float));

  return trimmed;
}

// Build cache position tensor from attention mask for KV cache management
// Used to track position in autoregressive generation
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

  // Fill the tensor with position values starting from pos0
  int64_t *p = cache_position.GetTensorMutableData<int64_t>();
  for (int32_t i = 0; i < seq_len; ++i) {
    p[i] = pos0 + i;
  }

  return cache_position;
}

}  // namespace

OfflineRecognizerQwen3ASRImpl::OfflineRecognizerQwen3ASRImpl(
    const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(config),
      config_(config),
      model_(std::make_unique<OfflineQwen3ASRModel>(config.model_config)),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          config.model_config.qwen3_asr.tokenizer)),
      rng_(config.model_config.qwen3_asr.seed) {
  InitFeatConfig();
}

template <typename Manager>
OfflineRecognizerQwen3ASRImpl::OfflineRecognizerQwen3ASRImpl(
    Manager *mgr, const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(mgr, config),
      config_(config),
      model_(std::make_unique<OfflineQwen3ASRModel>(mgr, config.model_config)),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          mgr, config.model_config.qwen3_asr.tokenizer)),
      rng_(config.model_config.qwen3_asr.seed) {
  InitFeatConfig();
}

std::unique_ptr<OfflineStream>
OfflineRecognizerQwen3ASRImpl::CreateStream() const {
  return std::make_unique<OfflineStream>(config_.feat_config);
}

void OfflineRecognizerQwen3ASRImpl::InitFeatConfig() {
  // Use Whisper-style feature extraction to match Python's WhisperFeatureExtractor
  config_.feat_config.feature_dim = 128;
  config_.feat_config.is_whisper = true;
  config_.feat_config.normalize_samples = true;
  config_.feat_config.snip_edges = false;
  config_.feat_config.dither = 0.0f;
}

std::vector<int64_t> OfflineRecognizerQwen3ASRImpl::BuildSourceIds(
    int32_t audio_token_len, int32_t &audio_beg_idx) const {
  const std::string system_text = "<|im_start|>system\n<|im_end|>\n";
  const std::string user_prefix = "<|im_start|>user\n<|audio_start|>";
  const std::string audio_pad = "<|audio_pad|>";
  const std::string user_suffix = "<|audio_end|><|im_end|>\n";
  const std::string assistant_text = "<|im_start|>assistant\n";

  std::vector<int64_t> ids_before = tokenizer_->Encode(system_text + user_prefix);
  std::vector<int64_t> audio_pad_ids = tokenizer_->Encode(audio_pad);
  std::vector<int64_t> ids_after = tokenizer_->Encode(user_suffix + assistant_text);

  audio_beg_idx = static_cast<int32_t>(ids_before.size());

  std::vector<int64_t> source_ids;
  size_t estimated_size = ids_before.size() +
                          static_cast<size_t>(audio_token_len) * audio_pad_ids.size() +
                          ids_after.size();
  source_ids.reserve(estimated_size);
  source_ids.insert(source_ids.end(), ids_before.begin(), ids_before.end());

  // Insert audio pad tokens (one <|audio_pad|> per audio token)
  for (int32_t i = 0; i < audio_token_len; ++i) {
    source_ids.insert(source_ids.end(), audio_pad_ids.begin(),
                     audio_pad_ids.end());
  }

  source_ids.insert(source_ids.end(), ids_after.begin(), ids_after.end());

  return source_ids;
}

// Sample token from logits with temperature and top-p sampling
int64_t OfflineRecognizerQwen3ASRImpl::SampleTokenFromLogitsFp16OrFp32(
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
      float v = p[i];
      if (std::isfinite(v) && v > best_val) {
        best_val = v;
        best = i;
        found_valid = true;
      }
    }
  }

  return found_valid ? best : 0;
}

// Sample token with temperature and top-p (nucleus) sampling
int64_t OfflineRecognizerQwen3ASRImpl::SampleTokenWithTemperatureAndTopP(
    const void *logits, bool is_fp16, int32_t vocab_size,
    float temperature, float top_p) const {
  // If temperature is 0 or very small, use argmax (greedy)
  if (temperature <= 1e-6f) {
    return SampleTokenFromLogitsFp16OrFp32(logits, is_fp16, vocab_size);
  }

  // Convert logits to probabilities
  std::vector<float> probs(vocab_size);
  if (is_fp16) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = HalfBitsToFloat(p[i]);
      probs[i] = std::isfinite(v) ? std::exp(v / temperature) : 0.0f;
    }
  } else {
    const float *p = reinterpret_cast<const float *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = p[i];
      probs[i] = std::isfinite(v) ? std::exp(v / temperature) : 0.0f;
    }
  }

  // Top-p (nucleus) sampling
  if (top_p < 1.0f - 1e-6f) {
    // Sort probabilities in descending order
    std::vector<std::pair<int32_t, float>> prob_idx;
    prob_idx.reserve(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) {
      prob_idx.push_back({i, probs[i]});
    }
    std::partial_sort(prob_idx.begin(), prob_idx.end(), prob_idx.end(),
                      [](const auto &a, const auto &b) {
                        return a.second > b.second;
                      });

    // Find cutoff cumulative probability
    float cumsum = 0.0f;
    int32_t cutoff = vocab_size;
    for (int32_t i = 0; i < vocab_size; ++i) {
      cumsum += prob_idx[i].second;
      if (cumsum >= top_p) {
        cutoff = i + 1;
        break;
      }
    }

    // Set probabilities after cutoff to 0
    for (int32_t i = cutoff; i < vocab_size; ++i) {
      prob_idx[i].second = 0.0f;
    }

    // Renormalize
    float sum = 0.0f;
    for (int32_t i = 0; i < cutoff; ++i) {
      sum += prob_idx[i].second;
    }
    if (sum > 0.0f) {
      for (int32_t i = 0; i < cutoff; ++i) {
        probs[prob_idx[i].first] = prob_idx[i].second / sum;
      }
    }

    // Sample from top-p
    float r = std::uniform_real_distribution<float>(0, 1)(rng_);
    cumsum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
      cumsum += probs[i];
      if (r <= cumsum) {
        return i;
      }
    }
    return vocab_size - 1;
  }

  // Simple sampling with temperature (no top-p)
  float sum = 0.0f;
  for (int32_t i = 0; i < vocab_size; ++i) {
    sum += probs[i];
  }
  if (sum > 0.0f) {
    for (int32_t i = 0; i < vocab_size; ++i) {
      probs[i] /= sum;
    }
  }

  float r = std::uniform_real_distribution<float>(0, 1)(rng_);
  float cumsum = 0.0f;
  for (int32_t i = 0; i < vocab_size; ++i) {
    cumsum += probs[i];
    if (r <= cumsum) {
      return i;
    }
  }
  return vocab_size - 1;
}

OfflineRecognitionResult OfflineRecognizerQwen3ASRImpl::GenerateText(
    Ort::Value audio_features, int32_t audio_token_len) const {
  OfflineRecognitionResult result;
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const auto &qwen3_config = config_.model_config.qwen3_asr;

  // Trim audio features to remove padding
  Ort::Value trimmed_audio_features =
      TrimAudioFeatures(std::move(audio_features), model_->Allocator());

  auto af_info = trimmed_audio_features.GetTensorTypeAndShapeInfo();
  auto af_shape = af_info.GetShape();

  int32_t hidden_size = static_cast<int32_t>(af_shape[2]);

  if (audio_token_len <= 0) {
    result.text = "";
    return result;
  }

  int32_t audio_beg_idx = 0;

  std::vector<int64_t> source_ids =
      BuildSourceIds(audio_token_len, audio_beg_idx);
  int32_t context_len = static_cast<int32_t>(source_ids.size());

  std::vector<std::pair<Ort::Value, Ort::Value>> cache_kv =
      model_->CreateEmptyKVCache(1);
  int32_t max_seq_len = model_->GetMaxTotalLen();

  if (context_len > max_seq_len) {
    source_ids.erase(source_ids.begin(), source_ids.end() - max_seq_len);
    audio_beg_idx = -1;
    context_len = static_cast<int32_t>(source_ids.size());
  }

  std::vector<int64_t> input_ids = source_ids;
  std::array<int64_t, 2> ids_shape{1, context_len};
  Ort::Value input_ids_tensor = Ort::Value::CreateTensor(
      memory_info, input_ids.data(), input_ids.size(), ids_shape.data(),
      ids_shape.size());

  std::array<int64_t, 2> attn_mask_shape{1, context_len};
  std::vector<int64_t> attn_mask_vec(context_len, 1);
  Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(
      memory_info, attn_mask_vec.data(), attn_mask_vec.size(),
      attn_mask_shape.data(), attn_mask_shape.size());

  Ort::Value cache_position = BuildCachePositionFromMask(
      attention_mask, context_len, model_->Allocator());

  // Create a view of trimmed_audio_features for the first forward pass
  // We need to keep it for subsequent decode steps
  Ort::Value audio_features_view = View(&trimmed_audio_features);

  auto tmp = model_->ForwardLLM(std::move(input_ids_tensor),
                                std::move(audio_features_view),
                                std::move(attention_mask), cache_position,
                                cache_kv);
  Ort::Value logits = std::move(tmp.first);
  auto kv_outputs = std::move(tmp.second);

  model_->ApplyKvDeltaInplace(&cache_kv, kv_outputs, cache_position);

  std::vector<int64_t> generated_ids;
  generated_ids.reserve(qwen3_config.max_new_tokens);

  const int64_t eos_id = tokenizer_->GetEosTokenId();
  const int32_t max_new_tokens = qwen3_config.max_new_tokens;

  // Sample first token from prefill logits
  auto log_info = logits.GetTensorTypeAndShapeInfo();
  auto log_shape = log_info.GetShape();

  if (log_shape.size() < 3) {
    result.text = "";
    return result;
  }

  // For prefill, logits shape is [batch, max_total_len, vocab_size]
  // but we only care about the first context_len positions
  int32_t time_dim = static_cast<int32_t>(log_shape[1]);
  int32_t vocab_size = static_cast<int32_t>(log_shape[2]);

  // Use context_len for prefill, not time_dim (which may be max_total_len)
  int32_t last_idx = context_len - 1;
  if (last_idx >= time_dim) {
    result.text = "";
    return result;
  }

  bool log_fp16 =
      (log_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  const void *base = nullptr;
  if (log_fp16)
    base = logits.GetTensorData<uint16_t>();
  else
    base = logits.GetTensorData<float>();

  const size_t offset = static_cast<size_t>(last_idx) * vocab_size;

  const void *last_logits =
      log_fp16 ? static_cast<const void *>(
                     reinterpret_cast<const uint16_t *>(base) + offset)
               : static_cast<const void *>(
                     reinterpret_cast<const float *>(base) + offset);

  int64_t next_id = SampleTokenWithTemperatureAndTopP(
      last_logits, log_fp16, vocab_size, qwen3_config.temperature,
      qwen3_config.top_p);

  if (next_id == eos_id) {
    result.text = "";
    return result;
  }

  generated_ids.push_back(next_id);
  int32_t cur_len = context_len + 1;

  // Decode loop: generate remaining tokens
  for (int32_t step = 1; step < max_new_tokens; ++step) {
    if (cur_len >= max_seq_len) break;

    // Decode step: forward single token
    int64_t last_token_id = next_id;
    std::vector<int64_t> one_id{last_token_id};
    std::array<int64_t, 2> one_shape{1, 1};
    Ort::Value one_tensor = Ort::Value::CreateTensor(
        memory_info, one_id.data(), one_id.size(), one_shape.data(),
        one_shape.size());

    // For decode step: attention_mask should be (1, 1) for single token input
    // Aligned with Python: attn_mask = np.ones((B, S), dtype=np.int64) where S=1
    std::array<int64_t, 2> mask_shape{1, 1};
    std::vector<int64_t> mask_vec(1, 1);
    Ort::Value next_attention_mask = Ort::Value::CreateTensor<int64_t>(
        memory_info, mask_vec.data(), mask_vec.size(), mask_shape.data(),
        mask_shape.size());

    // cache_position should be [cur_len] for the new token at position cur_len
    // Aligned with Python: cache_pos = np.arange(cur_len, cur_len + S, dtype=np.int64) where S=1
    std::array<int64_t, 1> cache_pos_shape{1};
    std::vector<int64_t> cache_pos_vec{static_cast<int64_t>(cur_len)};
    Ort::Value next_cache_position = Ort::Value::CreateTensor<int64_t>(
        memory_info, cache_pos_vec.data(), cache_pos_vec.size(),
        cache_pos_shape.data(), cache_pos_shape.size());

    // Create another view for decode step
    Ort::Value audio_features_view2 = View(&trimmed_audio_features);

    auto tmp2 = model_->ForwardLLM(std::move(one_tensor),
                                   std::move(audio_features_view2),
                                   std::move(next_attention_mask),
                                   next_cache_position, cache_kv);
    logits = std::move(tmp2.first);
    auto kv_outputs2 = std::move(tmp2.second);

    model_->ApplyKvDeltaInplace(&cache_kv, kv_outputs2, next_cache_position);

    // Sample next token from decode logits
    auto log_info2 = logits.GetTensorTypeAndShapeInfo();
    auto log_shape2 = log_info2.GetShape();

    if (log_shape2.size() < 3) {
      break;
    }

    int32_t time_dim2 = static_cast<int32_t>(log_shape2[1]);
    int32_t vocab_size2 = static_cast<int32_t>(log_shape2[2]);

    if (time_dim2 < 1) {
      break;
    }

    bool log_fp16_step =
        (log_info2.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

    const void *base2 = nullptr;
    if (log_fp16_step)
      base2 = logits.GetTensorData<uint16_t>();
    else
      base2 = logits.GetTensorData<float>();

    // Decode step: logits are at the last position
    int32_t step_last_idx = time_dim2 - 1;
    const size_t offset2 = static_cast<size_t>(step_last_idx) * vocab_size2;
    const void *last_logits2 =
        log_fp16_step ? static_cast<const void *>(
                       reinterpret_cast<const uint16_t *>(base2) + offset2)
                 : static_cast<const void *>(
                       reinterpret_cast<const float *>(base2) + offset2);

    next_id = SampleTokenWithTemperatureAndTopP(
        last_logits2, log_fp16_step, vocab_size2, qwen3_config.temperature,
        qwen3_config.top_p);

    if (next_id == eos_id) {
      break;
    }

    generated_ids.push_back(next_id);
    cur_len += 1;
  }

  result.text = tokenizer_->Decode(generated_ids);

  // Remove replacement characters
  result.text.erase(
      std::remove(result.text.begin(), result.text.end(), '\ufffd'),
      result.text.end());

  if (!generated_ids.empty()) {
    result.tokens.reserve(generated_ids.size());
    std::string pending_bytes;
    for (int64_t token_id : generated_ids) {
      std::string s =
          tokenizer_->GetTokenStringStreaming(token_id, &pending_bytes);
      result.tokens.push_back(std::move(s));
    }

    if (!pending_bytes.empty() && !result.tokens.empty()) {
      // Handle remaining bytes
      std::string replacement_chars;
      replacement_chars.reserve(pending_bytes.size() * 3);
      for (size_t i = 0; i < pending_bytes.size(); ++i) {
        replacement_chars.append("\xEF\xBF\xBD");
      }
      result.tokens.back().append(replacement_chars);
    }
  }

  return result;
}

void OfflineRecognizerQwen3ASRImpl::DecodeStreams(OfflineStream **ss,
                                                  int32_t n) const {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const auto &qwen3_config = config_.model_config.qwen3_asr;

  for (int32_t i = 0; i != n; ++i) {
    std::vector<float> f = ss[i]->GetFrames();

    // Apply Whisper-style normalization to match Python WhisperFeatureExtractor
    // Python uses log10 for mel spectrogram, then applies:
    //   log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    //   log_spec = (log_spec + 4.0) / 4.0
    // C++ returns mel energy (not log), need to apply log10 like Python
    if (config_.feat_config.is_whisper) {
      constexpr float kMinValue = 1e-10f;
      for (size_t j = 0; j < f.size(); ++j) {
        f[j] = log10f(f[j] + kMinValue);
      }

      // Find max value across all features (after log10 conversion)
      float max_val = f[0];
      for (size_t j = 1; j < f.size(); ++j) {
        if (f[j] > max_val) max_val = f[j];
      }

      // Apply: log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
      // and then: log_spec = (log_spec + 4.0) / 4.0
      float threshold = max_val - 8.0f;
      for (size_t j = 0; j < f.size(); ++j) {
        if (f[j] < threshold) {
          f[j] = threshold;
        }
        f[j] = (f[j] + 4.0f) / 4.0f;
      }
    }

    // Extract mel features: f is (num_frames * feature_dim) flattened
    int32_t num_frames = static_cast<int32_t>(
        f.size() / config_.feat_config.feature_dim);
    if (num_frames <= 0) {
      OfflineRecognitionResult r;
      r.text = "";
      ss[i]->SetResult(r);
      continue;
    }

    int32_t F = config_.feat_config.feature_dim;

    // GetFrames() returns flattened data in (T, F) layout:
    // [frame0_dim0, frame0_dim1, ..., frame0_dim(F-1), frame1_dim0, ...]
    // With is_whisper=true, GetFrames() returns Whisper-style log-mel spectrogram

    // Python uses 561 frames (num_frames-1), C++ has 562 frames
    // Use first 561 frames to match Python
    int32_t python_frames = num_frames - 1;

    std::array<int64_t, 3> conv_input_shape{1, static_cast<int64_t>(python_frames), static_cast<int64_t>(F)};

    Ort::Value conv_input = Ort::Value::CreateTensor<float>(
        memory_info, f.data(), static_cast<size_t>(python_frames) * F,
        conv_input_shape.data(), conv_input_shape.size());

    Ort::Value conv_output = model_->ForwardConvFrontend(std::move(conv_input));

    auto conv_info = conv_output.GetTensorTypeAndShapeInfo();
    auto conv_shape = conv_info.GetShape();

    int32_t conv_num_frames = static_cast<int32_t>(conv_shape[1]);

    // Create feature attention mask based on valid feature length
    // Calculate how many audio tokens we expect from the original feature length
    int32_t feat_len = python_frames;
    int32_t expected_audio_token_len = FeatToAudioTokensLen(feat_len, 100);

    // The conv_output shape[1] is the actual number of frames after conv
    // We need to create a mask that marks valid frames
    std::vector<int64_t> tok_mask_int64(conv_num_frames, 0);
    int32_t valid_frames = std::min(expected_audio_token_len, conv_num_frames);
    for (int32_t j = 0; j < valid_frames; ++j) {
      tok_mask_int64[j] = 1;
    }

    std::array<int64_t, 2> tok_mask_shape{1, conv_num_frames};
    Ort::Value feature_attention_mask = Ort::Value::CreateTensor<int64_t>(
        memory_info, tok_mask_int64.data(), tok_mask_int64.size(),
        tok_mask_shape.data(), tok_mask_shape.size());

    Ort::Value audio_features =
        model_->ForwardEncoder(std::move(conv_output),
                              std::move(feature_attention_mask));

    // Trim audio features before generating text
    Ort::Value trimmed_audio_features =
        TrimAudioFeatures(std::move(audio_features), model_->Allocator());

    // Use expected_audio_token_len for prompt (matches Python's a_len)
    OfflineRecognitionResult r =
        GenerateText(std::move(trimmed_audio_features), expected_audio_token_len);

    ss[i]->SetResult(r);
  }
}

#if __ANDROID_API__ >= 9
template OfflineRecognizerQwen3ASRImpl::OfflineRecognizerQwen3ASRImpl(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizerQwen3ASRImpl::OfflineRecognizerQwen3ASRImpl(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx
