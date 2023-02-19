// sherpa/csrc/online-transducer-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

class OnlineTransducerGreedySearchDecoder : public OnlineTransducerDecoder {
 public:
  explicit OnlineTransducerGreedySearchDecoder(OnlineTransducerModel *model)
      : model_(model) {}

  OnlineTransducerDecoderResult GetEmptyResult() override;

  void StripLeadingBlanks(OnlineTransducerDecoderResult *r) override;

  void Decode(Ort::Value encoder_out,
              std::vector<OnlineTransducerDecoderResult> *result) override;

 private:
  OnlineTransducerModel *model_;  // Not owned
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
