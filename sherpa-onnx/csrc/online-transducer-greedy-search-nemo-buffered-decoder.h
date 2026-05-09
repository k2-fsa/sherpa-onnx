// sherpa-onnx/csrc/online-transducer-greedy-search-nemo-buffered-decoder.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_BUFFERED_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_BUFFERED_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-buffered-model.h"

namespace sherpa_onnx {

class OnlineStream;

class OnlineTransducerGreedySearchNeMoBufferedDecoder {
 public:
  OnlineTransducerGreedySearchNeMoBufferedDecoder(
      OnlineTransducerNeMoBufferedModel *model, float blank_penalty)
      : model_(model), blank_penalty_(blank_penalty) {}

  void Decode(Ort::Value encoder_out, OnlineStream **ss, int32_t n) const;

 private:
  OnlineTransducerNeMoBufferedModel *model_;  // Not owned
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_BUFFERED_DECODER_H_
