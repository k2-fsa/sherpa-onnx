// sherpa-onnx/csrc/rknn/offline-ctc-greedy-search-decoder-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/offline-ctc-greedy-search-decoder-rknn.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"

namespace sherpa_onnx {

OfflineCtcDecoderResult OfflineCtcGreedySearchDecoderRknn::Decode(
    const float *logits, int32_t num_frames, int32_t vocab_size) {
  OfflineCtcDecoderResult ans;

  int64_t prev_id = -1;

  for (int32_t t = 0; t != num_frames; ++t) {
    int64_t y = MaxElementIndex(logits, vocab_size);

    logits += vocab_size;

    if (y != blank_id_ && y != prev_id) {
      ans.tokens.push_back(y);
      ans.timestamps.push_back(t);
    }
    prev_id = y;
  }  // for (int32_t t = 0; ...)

  return ans;
}

}  // namespace sherpa_onnx
