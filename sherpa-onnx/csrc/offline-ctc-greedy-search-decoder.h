// sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_CTC_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CTC_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-ctc-decoder.h"

namespace sherpa_onnx {

class OfflineCtcGreedySearchDecoder : public OfflineCtcDecoder {
 public:
  explicit OfflineCtcGreedySearchDecoder(int32_t blank_id)
      : blank_id_(blank_id) {}

  std::vector<OfflineCtcDecoderResult> Decode(
      Ort::Value log_probs, Ort::Value log_probs_length) override;

 private:
  int32_t blank_id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CTC_GREEDY_SEARCH_DECODER_H_
