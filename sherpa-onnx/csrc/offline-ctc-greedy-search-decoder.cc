// sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h"

#include <vector>

namespace sherpa_onnx {

std::vector<OfflineCtcDecoderResult> OfflineCtcGreedySearchDecoder::Decode(
    Ort::Value log_probs, Ort::Value log_probs_length) {
  return {};
}

}  // namespace sherpa_onnx
