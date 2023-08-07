// sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.h"

#include <algorithm>
#include <utility>

namespace sherpa_onnx {

std::vector<OfflineWhisperDecoderResult>
OfflineWhisperGreedySearchDecoder::Decode(Ort::Value cross_k,
                                          Ort::Value cross_v) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  auto self_kv_cache = model_->GetInitialSelfKVCache();

  std::vector<int64_t> initial_tokens = model_->GetInitialTokens();
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

  auto decoder_out = model_->ForwardDecoder(
      std::move(tokens), std::move(self_kv_cache.first),
      std::move(self_kv_cache.second), std::move(cross_k), std::move(cross_v),
      std::move(offset));

  const auto &logits = std::get<0>(decoder_out);
  const float *p_logits = logits.GetTensorData<float>();

  auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
  int32_t vocab_size = logits_shape[2];

  int32_t max_token_id = static_cast<int32_t>(std::distance(
      p_logits, std::max_element(p_logits, p_logits + vocab_size)));

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

    int64_t *p_offset =
        std::get<5>(decoder_out).GetTensorMutableData<int64_t>();

    if (i == 0) {
      *p_offset = initial_tokens.size();
    } else {
      *p_offset += 1;
    }

    decoder_out = model_->ForwardDecoder(std::move(tokens),
                                         std::move(std::get<1>(decoder_out)),
                                         std::move(std::get<2>(decoder_out)),
                                         std::move(std::get<3>(decoder_out)),
                                         std::move(std::get<4>(decoder_out)),
                                         std::move(std::get<5>(decoder_out)));

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
