// sherpa-onnx/csrc/offline-cohere-transcribe-greedy-search-decoder.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-cohere-transcribe-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

std::vector<OfflineCohereTranscribeDecoderResult>
OfflineCohereTranscribeGreedySearchDecoder::Decode(
    Ort::Value cross_k, Ort::Value cross_v, const std::vector<int64_t> &prompt,
    int32_t eos, int32_t num_feature_frames) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  auto shape = cross_k.GetTensorTypeAndShapeInfo().GetShape();
  if (shape[1] != 1) {
    SHERPA_ONNX_LOGE("This function supports only batch_size==1. Given: %d",
                     static_cast<int32_t>(shape[1]));
    return {};
  }

  int32_t batch_size = 1;
  std::array<int64_t, 2> token_shape{batch_size,
                                     static_cast<int64_t>(prompt.size())};

  Ort::Value tokens = Ort::Value::CreateTensor(
      memory_info, const_cast<int64_t *>(prompt.data()), prompt.size(),
      token_shape.data(), token_shape.size());

  int64_t offset_v = 0;
  Ort::Value offset =
      Ort::Value::CreateTensor<int64_t>(memory_info, &offset_v, 1, nullptr, 0);

  auto self_kv_cache = model_->GetInitialSelfKVCache();

  auto decoder_out =
      model_->ForwardDecoder(std::move(tokens), std::move(self_kv_cache.first),
                             std::move(self_kv_cache.second), View(&cross_k),
                             View(&cross_v), View(&offset));

  offset_v += prompt.size();

  auto logits_shape =
      std::get<0>(decoder_out).GetTensorTypeAndShapeInfo().GetShape();
  int32_t vocab_size = logits_shape[2];

  int32_t max_seq_len = model_->GetMaxSeqLen();

  const float *p_logits = std::get<0>(decoder_out).GetTensorData<float>();
  const float *p_start = p_logits + (logits_shape[1] - 1) * vocab_size;

  int64_t max_token_id = MaxElementIndex(p_start, vocab_size);

  std::vector<int32_t> predicted_tokens;

  // assume at most 6 tokens per second
  int32_t num_possible_tokens = num_feature_frames / 100.0 * 6;
  num_possible_tokens = std::min<int32_t>(num_possible_tokens, max_seq_len);

  token_shape = {1, 1};

  tokens = Ort::Value::CreateTensor(memory_info, &max_token_id, 1,
                                    token_shape.data(), token_shape.size());

  for (int32_t i = 0; i < num_possible_tokens; ++i) {
    if (max_token_id == eos) {
      break;
    }

    predicted_tokens.push_back(max_token_id);

    decoder_out = model_->ForwardDecoder(
        View(&tokens), std::move(std::get<1>(decoder_out)),
        std::move(std::get<2>(decoder_out)), View(&cross_k), View(&cross_v),
        View(&offset));

    offset_v += 1;

    const float *p_logits = std::get<0>(decoder_out).GetTensorData<float>();

    max_token_id = MaxElementIndex(p_logits, vocab_size);
  }

  std::vector<OfflineCohereTranscribeDecoderResult> ans(1);

  ans[0].tokens = std::move(predicted_tokens);

  return ans;
}

}  // namespace sherpa_onnx
