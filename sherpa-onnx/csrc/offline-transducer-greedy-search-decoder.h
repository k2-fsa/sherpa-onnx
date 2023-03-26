// sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-model.h"

namespace sherpa_onnx {

class OfflineTransducerGreedySearchDecoder : public OfflineTransducerDecoder {
 public:
  explicit OfflineTransducerGreedySearchDecoder(OfflineTransducerModel *model)
      : model_(model) {}

  std::vector<OfflineTransducerDecoderResult> Decode(
      Ort::Value encoder_out, Ort::Value encoder_out_length) override;

 private:
  OfflineTransducerModel *model_;  // Not owned
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
