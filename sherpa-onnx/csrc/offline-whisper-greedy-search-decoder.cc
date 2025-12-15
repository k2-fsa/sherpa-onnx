// sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
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

  initial_tokens.push_back(model_->NoTimeStampsToken());

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

  *(std::get<5>(decoder_out).GetTensorMutableData<int64_t>()) =
      initial_tokens.size();

  const auto &logits_tensor = std::get<0>(decoder_out);
  const float *p_logits = logits_tensor.GetTensorData<float>();

  auto logits_shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
  int32_t vocab_size = logits_shape[2];

  // Get logits for the first token
  const float *current_logits = p_logits + (logits_shape[1] - 1) * vocab_size;

  auto max_iter = std::max_element(current_logits, current_logits + vocab_size);
  float max_logit = *max_iter;
  int32_t current_token_id =
      static_cast<int32_t>(std::distance(current_logits, max_iter));

  int32_t n_text_ctx = model_->TextCtx();

  std::vector<int32_t> predicted_tokens;
  // Log probabilities.
  std::vector<float> predicted_log_probs;
  std::vector<std::vector<float>> predicted_vocab_log_probs;

  // assume at most 6 tokens per second
  int32_t num_possible_tokens = num_feature_frames / 100.0 * 6;
  num_possible_tokens = std::min<int32_t>(num_possible_tokens, n_text_ctx / 2);

  for (int32_t i = 0; i < num_possible_tokens; ++i) {
    if (current_token_id == model_->EOT()) {
      break;
    }

    // Calculate log-softmax for the full vocabulary
    // Allocate vector for this token (moved into result, so new allocation needed)
    // Reuse max_logit already found when determining current_token_id
    std::vector<float> full_vocab_probs(vocab_size);
    double sum_exp = 0.0;
    for (int32_t j = 0; j < vocab_size; ++j) {
      sum_exp += std::exp(current_logits[j] - max_logit);
    }
    // Calculate log_sum once (optimization: avoid recalculating log() in loop)
    float log_sum = max_logit + std::log(sum_exp);
    // Compute each probability
    for (int32_t j = 0; j < vocab_size; ++j) {
      full_vocab_probs[j] = current_logits[j] - log_sum;
    }

    // Extract log probability for the selected token
    float log_prob = full_vocab_probs[current_token_id];

    predicted_tokens.push_back(current_token_id);
    predicted_log_probs.push_back(log_prob);
    predicted_vocab_log_probs.push_back(std::move(full_vocab_probs));

    std::array<int64_t, 2> token_input_shape{1, 1};
    Ort::Value tokens_input = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), token_input_shape.data(),
        token_input_shape.size());
    int64_t *p_tokens = tokens_input.GetTensorMutableData<int64_t>();
    p_tokens[0] = current_token_id;

    decoder_out = model_->ForwardDecoder(std::move(tokens_input),
                                         std::move(std::get<1>(decoder_out)),
                                         std::move(std::get<2>(decoder_out)),
                                         std::move(std::get<3>(decoder_out)),
                                         std::move(std::get<4>(decoder_out)),
                                         std::move(std::get<5>(decoder_out)));

    int64_t *p_offset =
        std::get<5>(decoder_out).GetTensorMutableData<int64_t>();

    *p_offset += 1;
    if (*p_offset >= n_text_ctx - 1) {
      break;
    }

    const auto &next_logits_tensor = std::get<0>(decoder_out);
    current_logits = next_logits_tensor.GetTensorData<float>();

    auto next_max_iter =
        std::max_element(current_logits, current_logits + vocab_size);
    max_logit = *next_max_iter;
    current_token_id =
        static_cast<int32_t>(std::distance(current_logits, next_max_iter));
  }

  std::vector<OfflineWhisperDecoderResult> ans(1);

  const auto &id2lang = model_->GetID2Lang();
  if (model_->IsMultiLingual() && id2lang.count(initial_tokens[1])) {
    ans[0].lang = id2lang.at(initial_tokens[1]);
  } else {
    ans[0].lang = "";
  }

  ans[0].tokens = std::move(predicted_tokens);
  ans[0].token_log_probs = std::move(predicted_log_probs);
  ans[0].vocab_log_probs = std::move(predicted_vocab_log_probs);

  return ans;
}

}  // namespace sherpa_onnx
