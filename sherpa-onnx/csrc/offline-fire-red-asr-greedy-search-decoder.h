// sherpa-onnx/csrc/offline-fire-red-asr-greedy-search-decoder.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-fire-red-asr-decoder.h"
#include "sherpa-onnx/csrc/offline-fire-red-asr-model.h"

namespace sherpa_onnx {

class OfflineFireRedAsrGreedySearchDecoder : public OfflineFireRedAsrDecoder {
 public:
  explicit OfflineFireRedAsrGreedySearchDecoder(OfflineFireRedAsrModel *model)
      : model_(model) {}

  std::vector<OfflineFireRedAsrDecoderResult> Decode(
      Ort::Value cross_k, Ort::Value cross_v) override;

 private:
  OfflineFireRedAsrModel *model_;  // not owned
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_GREEDY_SEARCH_DECODER_H_
