// sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.h"

#include <algorithm>
#include <vector>

namespace sherpa_onnx {

std::vector<OfflineParaformerDecoderResult>
OfflineParaformerGreedySearchDecoder::Decode(
    Ort::Value /*log_probs*/, Ort::Value token_num,
    Ort::Value us_cif_peak /*=Ort::Value(nullptr)*/
) {
  std::vector<int64_t> shape = token_num.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = shape[0];
  int32_t max_num_tokens = shape[1];

  std::vector<OfflineParaformerDecoderResult> results(batch_size);

  if (!us_cif_peak) {
    // when timestamp is enabled, the data type of token_num is int32_t
    const int64_t *p_token = token_num.GetTensorData<int64_t>();

    for (int32_t i = 0; i != batch_size; ++i, p_token += max_num_tokens) {
      for (int32_t k = 0; k != max_num_tokens; ++k) {
        int32_t t = p_token[k];
        if (t == eos_id_) {
          break;
        }

        results[i].tokens.push_back(t);
      }
    }
  }

  return results;
}

}  // namespace sherpa_onnx
