// sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-whisper-timestamp-rules.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

void OfflineWhisperGreedySearchDecoder::SetConfig(
    const OfflineWhisperModelConfig &config) {
  config_ = config;
}

std::vector<OfflineWhisperDecoderResult>
OfflineWhisperGreedySearchDecoder::Decode(Ort::Value cross_k,
                                          Ort::Value cross_v,
                                          int32_t num_feature_frames) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // Check if we should collect attention weights for DTW timestamp computation
  bool collect_attention = config_.enable_timestamps &&
                           model_->HasAttentionOutput();

  // Warn once if timestamps requested but model doesn't support it
  static bool warned_no_attention = false;
  if (config_.enable_timestamps && !model_->HasAttentionOutput() &&
      !warned_no_attention) {
    warned_no_attention = true;
    SHERPA_ONNX_LOGE(
        "Warning: enable_timestamps=true but the decoder model does not have "
        "cross-attention outputs. Timestamps will not be available. "
        "To enable timestamps, export the model with attention outputs using: "
        "python scripts/whisper/export-onnx-with-attention.py");
  }

  // For multilingual models, initial_tokens contains [sot, language, task]
  //   - language is English by default
  //   - task is transcribe by default
  //
  // For non-multilingual models, initial_tokens contains [sot]
  std::vector<int64_t> initial_tokens = model_->GetInitialTokens();

  if (model_->IsMultiLingual()) {
    if (!config_.language.empty()) {
      const auto &lang2id = model_->GetLang2ID();

      if (!lang2id.count(config_.language)) {
        SHERPA_ONNX_LOGE("Invalid language: %s", config_.language.c_str());
        exit(-1);
      }

      int32_t lang_id = lang2id.at(config_.language);

      // 0: sot, 1: lang_id, 2: task, 3: no_timestamps
      initial_tokens[1] = lang_id;
    } else {
      int32_t lang_id = model_->DetectLanguage(cross_k, cross_v);

      // 0: sot, 1: lang_id, 2: task, 3: no_timestamps
      initial_tokens[1] = lang_id;
    }

    if (config_.task == "translate") {
      initial_tokens[2] = model_->Translate();
    } else if (config_.task != "transcribe") {
      // initial_tokens[2] is transcribe by default
      SHERPA_ONNX_LOGE(
          "Unsupported task: %s. Valid values are: transcribe, translate.",
          config_.task.c_str());
    }
  }

  // Add no_timestamps token when NOT using segment timestamp mode.
  // When enable_segment_timestamps=true, we let the decoder output timestamp
  // tokens (like <|0.00|>) which serve as alignment anchors.
  // When enable_timestamps=true (DTW mode), we MUST include no_timestamps
  // because OpenAI's alignment (timing.py) uses it as an anchor token at the
  // start of the DTW matrix. Without it, the first text token is misaligned.
  if (!config_.enable_segment_timestamps) {
    initial_tokens.push_back(model_->NoTimeStampsToken());
  }

  // Track if we're using segment timestamp mode
  bool enable_segment_timestamps = config_.enable_segment_timestamps;

  // Get token IDs for timestamp rules
  int32_t timestamp_begin = model_->TimestampBegin();
  int32_t no_timestamps = model_->NoTimeStampsToken();
  int32_t eot = model_->EOT();

  // Max initial timestamp: 50 = 1.0 second (each timestamp is 0.02s)
  constexpr int32_t kMaxInitialTimestampIndex = 50;

  // Maintain running list of all tokens for timestamp rules
  std::vector<int64_t> all_tokens = initial_tokens;
  int32_t sample_begin = static_cast<int32_t>(initial_tokens.size());

  int32_t batch_size = 1;
  std::array<int64_t, 2> token_shape{
      batch_size, static_cast<int64_t>(initial_tokens.size())};

  Ort::Value tokens = Ort::Value::CreateTensor(
      memory_info, initial_tokens.data(), initial_tokens.size(),
      token_shape.data(), token_shape.size());

  std::array<int64_t, 1> offset_shape{1};
  Ort::Value offset = Ort::Value::CreateTensor<int64_t>(
      model_->Allocator(), offset_shape.data(), offset_shape.size());
  *(offset.GetTensorMutableData<int64_t>()) = 0;

  auto self_kv_cache = model_->GetInitialSelfKVCache();

  auto decoder_out = model_->ForwardDecoder(
      std::move(tokens), std::move(self_kv_cache.first),
      std::move(self_kv_cache.second), std::move(cross_k), std::move(cross_v),
      std::move(offset));

  // Note: decoder_out is now a 7-tuple with attention weights as 7th element
  // Indices: 0=logits, 1=self_k, 2=self_v, 3=cross_k, 4=cross_v, 5=offset, 6=attention
  *(std::get<5>(decoder_out).GetTensorMutableData<int64_t>()) =
      initial_tokens.size();

  auto logits_shape = std::get<0>(decoder_out).GetTensorTypeAndShapeInfo().GetShape();
  int32_t vocab_size = logits_shape[2];

  int32_t n_text_ctx = model_->TextCtx();
  int32_t max_token_id = 0;

  // Get initial logits
  {
    const float *p_logits = std::get<0>(decoder_out).GetTensorData<float>();
    const float *p_start = p_logits + (logits_shape[1] - 1) * vocab_size;

    if (enable_segment_timestamps) {
      // Make a copy of logits for applying timestamp rules
      std::vector<float> logits_copy(p_start, p_start + vocab_size);
      ApplyTimestampRules(logits_copy.data(), vocab_size, all_tokens,
                          sample_begin, timestamp_begin, no_timestamps, eot,
                          kMaxInitialTimestampIndex);
      auto max_it =
          std::max_element(logits_copy.begin(), logits_copy.end());
      max_token_id = static_cast<int32_t>(std::distance(logits_copy.begin(), max_it));
    } else {
      auto max_it = std::max_element(p_start, p_start + vocab_size);
      max_token_id = static_cast<int32_t>(std::distance(p_start, max_it));
    }
  }

  std::vector<int32_t> predicted_tokens;

  // Storage for accumulated attention weights
  std::vector<std::vector<float>> all_attention_weights;
  int32_t attention_n_heads = 0;
  int32_t attention_n_frames = 0;

  // Track indices of timestamp tokens in the attention sequence
  // (0-based, relative to the start of all_attention_weights)
  std::vector<int32_t> timestamp_token_indices;

  // Collect attention from initial tokens if enabled
  if (collect_attention) {
    auto &attn = std::get<6>(decoder_out);
    auto attn_shape = attn.GetTensorTypeAndShapeInfo().GetShape();
    // Shape: (batch, n_heads, n_tokens, n_audio_ctx)
    if (attn_shape.size() >= 4 && attn_shape[1] > 0) {
      attention_n_heads = static_cast<int32_t>(attn_shape[1]);
      attention_n_frames = static_cast<int32_t>(attn_shape[3]);
      int32_t n_initial_tokens = static_cast<int32_t>(attn_shape[2]);

      const float *p_attn = attn.GetTensorData<float>();
      int32_t stride = attention_n_frames;

      // Store attention for each initial token
      for (int32_t t = 0; t < n_initial_tokens; ++t) {
        std::vector<float> token_attn(attention_n_heads * attention_n_frames);
        for (int32_t h = 0; h < attention_n_heads; ++h) {
          const float *src = p_attn + h * n_initial_tokens * stride + t * stride;
          std::copy(src, src + attention_n_frames,
                    token_attn.begin() + h * attention_n_frames);
        }
        all_attention_weights.push_back(std::move(token_attn));
      }
    }
  }

  // assume at most 6 tokens per second
  int32_t num_possible_tokens = num_feature_frames / 100.0 * 6;
  num_possible_tokens = std::min<int32_t>(num_possible_tokens, n_text_ctx / 2);

  for (int32_t i = 0; i < num_possible_tokens; ++i) {
    if (max_token_id == eot) {
      break;
    }

    predicted_tokens.push_back(max_token_id);
    all_tokens.push_back(max_token_id);

    // Track if this is a timestamp token (for filtering in DTW)
    if (max_token_id >= timestamp_begin) {
      // The attention index is: initial_tokens.size() + current predicted index
      int32_t attn_idx = static_cast<int32_t>(initial_tokens.size()) +
                         static_cast<int32_t>(predicted_tokens.size()) - 1;
      timestamp_token_indices.push_back(attn_idx);
    }

    std::array<int64_t, 2> token_shape{1, 1};
    Ort::Value tokens = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), token_shape.data(), token_shape.size());

    int64_t *p_tokens = tokens.GetTensorMutableData<int64_t>();
    p_tokens[0] = max_token_id;

    decoder_out = model_->ForwardDecoder(std::move(tokens),
                                         std::move(std::get<1>(decoder_out)),
                                         std::move(std::get<2>(decoder_out)),
                                         std::move(std::get<3>(decoder_out)),
                                         std::move(std::get<4>(decoder_out)),
                                         std::move(std::get<5>(decoder_out)));

    // Collect attention for this token
    if (collect_attention) {
      auto &attn = std::get<6>(decoder_out);
      auto attn_shape = attn.GetTensorTypeAndShapeInfo().GetShape();
      if (attn_shape.size() >= 4 && attn_shape[1] == attention_n_heads) {
        const float *p_attn = attn.GetTensorData<float>();
        // Shape: (batch, n_heads, 1, n_audio_ctx) - single token
        std::vector<float> token_attn(attention_n_heads * attention_n_frames);
        for (int32_t h = 0; h < attention_n_heads; ++h) {
          const float *src = p_attn + h * attention_n_frames;
          std::copy(src, src + attention_n_frames,
                    token_attn.begin() + h * attention_n_frames);
        }
        all_attention_weights.push_back(std::move(token_attn));
      }
    }

    int64_t *p_offset =
        std::get<5>(decoder_out).GetTensorMutableData<int64_t>();

    *p_offset += 1;
    if (*p_offset >= n_text_ctx - 1) {
      break;
    }

    const float *p_logits = std::get<0>(decoder_out).GetTensorData<float>();

    if (enable_segment_timestamps) {
      // Make a copy of logits for applying timestamp rules
      std::vector<float> logits_copy(p_logits, p_logits + vocab_size);
      // After first token, don't apply max_initial_timestamp constraint
      ApplyTimestampRules(logits_copy.data(), vocab_size, all_tokens,
                          sample_begin, timestamp_begin, no_timestamps, eot,
                          -1);  // -1 = no max_initial constraint
      auto max_it =
          std::max_element(logits_copy.begin(), logits_copy.end());
      max_token_id = static_cast<int32_t>(std::distance(logits_copy.begin(), max_it));

    } else {
      auto max_it = std::max_element(p_logits, p_logits + vocab_size);
      max_token_id = static_cast<int32_t>(std::distance(p_logits, max_it));
    }
  }

  std::vector<OfflineWhisperDecoderResult> ans(1);

  const auto &id2lang = model_->GetID2Lang();
  if (id2lang.count(initial_tokens[1])) {
    ans[0].lang = id2lang.at(initial_tokens[1]);
  } else {
    ans[0].lang = "";
  }

  ans[0].tokens = std::move(predicted_tokens);

  // Parse timestamp tokens into segments if using segment timestamp mode
  if (enable_segment_timestamps) {
    ans[0].segments = ParseTimestampTokens(ans[0].tokens, timestamp_begin, eot);
  }

  // Add accumulated attention weights if available
  if (collect_attention && !all_attention_weights.empty()) {
    int32_t n_tokens = static_cast<int32_t>(all_attention_weights.size());
    ans[0].attention_n_heads = attention_n_heads;
    ans[0].attention_n_tokens = n_tokens;
    ans[0].attention_n_frames = attention_n_frames;
    // Actual audio frames for clipping (encoder downsamples by factor of 2)
    ans[0].num_audio_frames = num_feature_frames / 2;

    // Flatten to (n_heads, n_tokens, n_frames)
    ans[0].attention_weights.resize(attention_n_heads * n_tokens * attention_n_frames);
    for (int32_t h = 0; h < attention_n_heads; ++h) {
      for (int32_t t = 0; t < n_tokens; ++t) {
        const float *src = all_attention_weights[t].data() + h * attention_n_frames;
        float *dst = ans[0].attention_weights.data() +
                     h * n_tokens * attention_n_frames + t * attention_n_frames;
        std::copy(src, src + attention_n_frames, dst);
      }
    }

    // Add timestamp token indices for DTW filtering
    ans[0].timestamp_token_indices = std::move(timestamp_token_indices);
  }

  return ans;
}

}  // namespace sherpa_onnx
