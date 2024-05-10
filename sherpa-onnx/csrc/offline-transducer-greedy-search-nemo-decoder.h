// sherpa-onnx/csrc/offline-transducer-greedy-search-nemo-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-nemo-model.h"

namespace sherpa_onnx {

class OfflineTransducerGreedySearchNeMoDecoder
    : public OfflineTransducerDecoder {
 public:
  OfflineTransducerGreedySearchNeMoDecoder(OfflineTransducerNeMoModel *model,
                                           float blank_penalty)
      : model_(model), blank_penalty_(blank_penalty) {}

  std::vector<OfflineTransducerDecoderResult> Decode(
      Ort::Value encoder_out, Ort::Value encoder_out_length,
      OfflineStream **ss = nullptr, int32_t n = 0) override;

 private:
  OfflineTransducerNeMoModel *model_;  // Not owned
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_NEMO_DECODER_H_
