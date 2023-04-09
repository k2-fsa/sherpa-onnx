// sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.h"

#include <vector>

#include "sherpa-onnx/csrc/math.h"

namespace sherpa_onnx {

std::vector<OfflineParaformerDecoderResult>
OfflineParaformerGreedySearchDecoder::Decode(Ort::Value log_probs,
                                             Ort::Value /*token_num*/) {
  std::vector<int64_t> shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = shape[0];
  int32_t num_tokens = shape[1];
  int32_t vocab_size = shape[2];

  std::vector<OfflineParaformerDecoderResult> results(batch_size);

  const float *p = log_probs.GetTensorData<float>();
  for (int32_t i = 0; i != batch_size; ++i) {
    for (int32_t k = 0; k != num_tokens; ++k) {
      int32_t max_idx = ArgMax(p, vocab_size);
      if (max_idx == eos_id_) break;

      results[i].tokens.push_back(max_idx);

      p += vocab_size;
    }
  }

  return results;
}

}  // namespace sherpa_onnx
