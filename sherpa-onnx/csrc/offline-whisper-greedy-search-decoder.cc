// sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.h"

#include <algorithm>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

int32_t OfflineWhisperGreedySearchDecoder::DetectLanguage(
    Ort::Value &cross_k, Ort::Value &cross_v) const {  // NOLINT
  int64_t token_val = model_->SOT();
  std::array<int64_t, 2> token_shape{1, 1};

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  Ort::Value tokens = Ort::Value::CreateTensor(
      memory_info, &token_val, 1, token_shape.data(), token_shape.size());

  auto self_kv_cache = model_->GetInitialSelfKVCache();

  std::array<int64_t, 1> offset_shape{1};
  Ort::Value offset = Ort::Value::CreateTensor<int64_t>(
      model_->Allocator(), offset_shape.data(), offset_shape.size());
  *(offset.GetTensorMutableData<int64_t>()) = 0;

  auto decoder_out = model_->ForwardDecoder(
      std::move(tokens), std::move(self_kv_cache.first),
      std::move(self_kv_cache.second), std::move(cross_k), std::move(cross_v),
      std::move(offset));

  cross_k = std::move(std::get<3>(decoder_out));
  cross_v = std::move(std::get<4>(decoder_out));

  const float *p_logits = std::get<0>(decoder_out).GetTensorData<float>();
  int32_t vocab_size = model_->VocabSize();
  const auto &all_language_ids = model_->GetAllLanguageIDs();

  int32_t lang_id = all_language_ids[0];
  float this_logit = p_logits[lang_id];

  for (int32_t i = 1; i != all_language_ids.size(); ++i) {
    int32_t id = all_language_ids[i];
    float p = p_logits[id];

    if (p > this_logit) {
      this_logit = p;
      lang_id = id;
    }
  }
#if 1
  SHERPA_ONNX_LOGE("Detected language: %s",
                   model_->GetID2Lang().at(lang_id).c_str());
#endif

  return lang_id;
}

std::vector<OfflineWhisperDecoderResult>
OfflineWhisperGreedySearchDecoder::Decode(Ort::Value cross_k,
                                          Ort::Value cross_v) {
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
      int32_t lang_id = DetectLanguage(cross_k, cross_v);

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

  const auto &logits = std::get<0>(decoder_out);
  const float *p_logits = logits.GetTensorData<float>();

  auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
  int32_t vocab_size = logits_shape[2];

  const float *p_start = p_logits + (logits_shape[1] - 1) * vocab_size;

  int32_t max_token_id = static_cast<int32_t>(
      std::distance(p_start, std::max_element(p_start, p_start + vocab_size)));

  int32_t n_text_ctx = model_->TextCtx();

  std::vector<int32_t> predicted_tokens;
  for (int32_t i = 0; i < n_text_ctx; ++i) {
    if (max_token_id == model_->EOT()) {
      break;
    }

    predicted_tokens.push_back(max_token_id);

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

    int64_t *p_offset =
        std::get<5>(decoder_out).GetTensorMutableData<int64_t>();

    *p_offset += 1;

    const auto &logits = std::get<0>(decoder_out);
    const float *p_logits = logits.GetTensorData<float>();

    max_token_id = static_cast<int64_t>(std::distance(
        p_logits, std::max_element(p_logits, p_logits + vocab_size)));
  }

  std::vector<OfflineWhisperDecoderResult> ans(1);

  ans[0].tokens = std::move(predicted_tokens);

  return ans;
}

}  // namespace sherpa_onnx
