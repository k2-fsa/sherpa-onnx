// sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h"

namespace sherpa_onnx {

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerGreedySearchDecoder::Decode(Ort::Value encoder_out,
                                             Ort::Value encoder_out_length) {
  return {};
}

}  // namespace sherpa_onnx
