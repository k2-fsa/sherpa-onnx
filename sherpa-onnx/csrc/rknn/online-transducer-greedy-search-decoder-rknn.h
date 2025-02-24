// sherpa-onnx/csrc/rknn/online-transducer-greedy-search-decoder-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_RKNN_H_

#include <vector>

#include "sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.h"

namespace sherpa_onnx {

struct OnlineTransducerDecoderResultRknn {
  /// Number of frames after subsampling we have decoded so far
  int32_t frame_offset = 0;

  /// The decoded token IDs so far
  std::vector<int64_t> tokens;

  /// number of trailing blank frames decoded so far
  int32_t num_trailing_blanks = 0;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;

  std::vector<float> previous_decoder_out;
};

class OnlineTransducerGreedySearchDecoderRknn {
 public:
  explicit OnlineTransducerGreedySearchDecoderRknn(
      OnlineZipformerTransducerModelRknn *model, int32_t unk_id = 2,
      float blank_penalty = 0.0)
      : model_(model), unk_id_(unk_id), blank_penalty_(blank_penalty) {}

  OnlineTransducerDecoderResultRknn GetEmptyResult() const;

  void StripLeadingBlanks(OnlineTransducerDecoderResultRknn *r) const;

  void Decode(std::vector<float> encoder_out,
              OnlineTransducerDecoderResultRknn *result) const;

 private:
  OnlineZipformerTransducerModelRknn *model_;  // Not owned
  int32_t unk_id_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_RKNN_H_
