// sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

std::vector<OfflineParaformerDecoderResult>
OfflineParaformerGreedySearchDecoder::Decode(
    Ort::Value log_probs, Ort::Value /*token_num*/,
    Ort::Value us_cif_peak /*=Ort::Value(nullptr)*/
) {
  std::vector<int64_t> shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = shape[0];
  int32_t num_tokens = shape[1];
  int32_t vocab_size = shape[2];

  std::vector<OfflineParaformerDecoderResult> results(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *p =
        log_probs.GetTensorData<float>() + i * num_tokens * vocab_size;
    for (int32_t k = 0; k != num_tokens; ++k) {
      auto max_idx = static_cast<int64_t>(
          std::distance(p, std::max_element(p, p + vocab_size)));
      if (max_idx == eos_id_) {
        break;
      }

      results[i].tokens.push_back(max_idx);

      p += vocab_size;
    }

    if (us_cif_peak) {
      int32_t dim = us_cif_peak.GetTensorTypeAndShapeInfo().GetShape().back();

      const auto *peak = us_cif_peak.GetTensorData<float>() + i * dim;
      std::vector<float> timestamps;
      timestamps.reserve(results[i].tokens.size());

      // 10.0: frameshift is 10 milliseconds
      // 6: LfrWindowSize
      // 3: us_cif_peak is upsampled by a factor of 3
      // 1000: milliseconds to seconds
      float scale = 10.0 * 6 / 3 / 1000;

      for (int32_t k = 0; k != dim; ++k) {
        if (peak[k] > 1 - 1e-4) {
          timestamps.push_back(k * scale);
        }
      }

      if (!timestamps.empty()) {
        timestamps.pop_back();
      }

      if (timestamps.size() == results[i].tokens.size()) {
        results[i].timestamps = std::move(timestamps);
      }
    }
  }

  return results;
}

}  // namespace sherpa_onnx
