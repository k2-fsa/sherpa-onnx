// sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_

#include <vector>
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-model.h"

namespace sherpa_onnx {

class OnlineTransducerGreedySearchNeMoDecoder : public OnlineTransducerDecoder {
 public:
  OnlineTransducerGreedySearchNeMoDecoder(OnlineTransducerNeMoModel *model,
                                          float blank_penalty);

  std::vector<OnlineTransducerDecoderResult> Decode(
    Ort::Value encoder_out, 
    Ort::Value encoder_out_length, 
    std::vector<Ort::Value> decoder_states, 
    std::vector<OnlineTransducerDecoderResult> *results,
    OnlineStream **ss = nullptr, int32_t n = 0);

 private:
  OnlineTransducerDecoderResult DecodeChunkByChunk(const float *encoder_out,
                                                   int32_t num_rows,
                                                   int32_t num_cols);

  OnlineTransducerNeMoModel *model_;  // Not owned
  float blank_penalty_;
  // std::vector<Ort::Value> decoder_states_;  // Decoder states to be maintained across chunks
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_

