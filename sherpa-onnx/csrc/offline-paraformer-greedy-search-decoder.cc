// sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.h"

#include <vector>

namespace sherpa_onnx {

std::vector<OfflineParaformerDecoderResult>
OfflineParaformerGreedySearchDecoder::Decode(Ort::Value /*log_probs*/,
                                             Ort::Value token_num) {
  std::vector<int64_t> shape = token_num.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = shape[0];
  int32_t num_tokens = shape[1];

  std::vector<OfflineParaformerDecoderResult> results(batch_size);

  const int64_t *p = token_num.GetTensorData<int64_t>();
  for (int32_t i = 0; i != batch_size; ++i) {
    for (int32_t k = 0; k != num_tokens; ++k) {
      if (p[k] == eos_id_) break;

      results[i].tokens.push_back(p[k]);
    }

    p += num_tokens;
  }

  return results;
}

}  // namespace sherpa_onnx
