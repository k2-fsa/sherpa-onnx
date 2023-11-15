// sherpa-onnx/csrc/online-ctc-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/online-ctc-decoder.h"

namespace sherpa_onnx {

class OnlineCtcGreedySearchDecoder : public OnlineCtcDecoder {
 public:
  explicit OnlineCtcGreedySearchDecoder(int32_t blank_id)
      : blank_id_(blank_id) {}

  void Decode(Ort::Value log_probs,
              std::vector<OnlineCtcDecoderResult> *results) override;

 private:
  int32_t blank_id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_GREEDY_SEARCH_DECODER_H_
