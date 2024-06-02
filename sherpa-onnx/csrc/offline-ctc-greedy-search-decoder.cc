// sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

std::vector<OfflineCtcDecoderResult> OfflineCtcGreedySearchDecoder::Decode(
    Ort::Value log_probs, Ort::Value log_probs_length) {
  std::vector<int64_t> shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t num_frames = static_cast<int32_t>(shape[1]);
  int32_t vocab_size = static_cast<int32_t>(shape[2]);

  const int64_t *p_log_probs_length = log_probs_length.GetTensorData<int64_t>();

  std::vector<OfflineCtcDecoderResult> ans;
  ans.reserve(batch_size);

  for (int32_t b = 0; b != batch_size; ++b) {
    const float *p_log_probs =
        log_probs.GetTensorData<float>() + b * num_frames * vocab_size;

    OfflineCtcDecoderResult r;
    int64_t prev_id = -1;

    for (int32_t t = 0; t != static_cast<int32_t>(p_log_probs_length[b]); ++t) {
      auto y = static_cast<int64_t>(std::distance(
          static_cast<const float *>(p_log_probs),
          std::max_element(
              static_cast<const float *>(p_log_probs),
              static_cast<const float *>(p_log_probs) + vocab_size)));
      p_log_probs += vocab_size;

      if (y != blank_id_ && y != prev_id) {
        r.tokens.push_back(y);
        r.timestamps.push_back(t);
      }
      prev_id = y;
    }  // for (int32_t t = 0; ...)

    ans.push_back(std::move(r));
  }
  return ans;
}

}  // namespace sherpa_onnx
