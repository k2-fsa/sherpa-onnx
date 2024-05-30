// sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_

#include <vector>
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-model.h"

namespace sherpa_onnx {

class OnlineTransducerGreedySearchNeMoDecoder {
 public:
  OnlineTransducerGreedySearchNeMoDecoder(OnlineTransducerNeMoModel *model,
                                          float blank_penalty)
      : model_(model),
      blank_penalty_(blank_penalty) {}

  OnlineTransducerDecoderResult GetEmptyResult() const;
  void UpdateDecoderOut(OnlineTransducerDecoderResult *result) {}
  void StripLeadingBlanks(OnlineTransducerDecoderResult * /*r*/) const {}
  
  std::vector<Ort::Value> Decode(
    Ort::Value encoder_out, 
    std::vector<Ort::Value> decoder_states, 
    std::vector<OnlineTransducerDecoderResult> *result,
    OnlineStream **ss = nullptr, int32_t n = 0);

 private:
  OnlineTransducerNeMoModel *model_;  // Not owned
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_

