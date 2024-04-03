// sherpa-onnx/csrc/online-ctc-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-ctc-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineCtcGreedySearchDecoder::Decode(
    Ort::Value log_probs, std::vector<OnlineCtcDecoderResult> *results,
    OnlineStream ** /*ss=nullptr*/, int32_t /*n = 0*/) {
  std::vector<int64_t> log_probs_shape =
      log_probs.GetTensorTypeAndShapeInfo().GetShape();

  if (log_probs_shape[0] != results->size()) {
    SHERPA_ONNX_LOGE("Size mismatch! log_probs.size(0) %d, results.size(0): %d",
                     static_cast<int32_t>(log_probs_shape[0]),
                     static_cast<int32_t>(results->size()));
    exit(-1);
  }

  int32_t batch_size = static_cast<int32_t>(log_probs_shape[0]);
  int32_t num_frames = static_cast<int32_t>(log_probs_shape[1]);
  int32_t vocab_size = static_cast<int32_t>(log_probs_shape[2]);

  const float *p = log_probs.GetTensorData<float>();

  for (int32_t b = 0; b != batch_size; ++b) {
    auto &r = (*results)[b];

    int32_t prev_id = -1;

    for (int32_t t = 0; t != num_frames; ++t, p += vocab_size) {
      int32_t y = static_cast<int32_t>(std::distance(
          static_cast<const float *>(p),
          std::max_element(static_cast<const float *>(p),
                           static_cast<const float *>(p) + vocab_size)));

      if (y == blank_id_) {
        r.num_trailing_blanks += 1;
      } else {
        r.num_trailing_blanks = 0;
      }

      if (y != blank_id_ && y != prev_id) {
        r.tokens.push_back(y);
        r.timestamps.push_back(t + r.frame_offset);
      }

      prev_id = y;
    }  // for (int32_t t = 0; t != num_frames; ++t) {
  }    // for (int32_t b = 0; b != batch_size; ++b)

  // Update frame_offset
  for (auto &r : *results) {
    r.frame_offset += num_frames;
  }
}

}  // namespace sherpa_onnx
