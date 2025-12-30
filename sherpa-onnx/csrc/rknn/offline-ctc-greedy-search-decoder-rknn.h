// sherpa-onnx/csrc/rknn/offline-ctc-greedy-search-decoder-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_OFFLINE_CTC_GREEDY_SEARCH_DECODER_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_OFFLINE_CTC_GREEDY_SEARCH_DECODER_RKNN_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-ctc-decoder.h"

namespace sherpa_onnx {

class OfflineCtcGreedySearchDecoderRknn {
 public:
  explicit OfflineCtcGreedySearchDecoderRknn(int32_t blank_id)
      : blank_id_(blank_id) {}

  OfflineCtcDecoderResult Decode(const float *logits, int32_t num_frames,
                                 int32_t vocab_size);

 private:
  int32_t blank_id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_OFFLINE_CTC_GREEDY_SEARCH_DECODER_RKNN_H_
