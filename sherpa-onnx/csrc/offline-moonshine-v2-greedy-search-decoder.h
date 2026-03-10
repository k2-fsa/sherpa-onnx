// sherpa-onnx/csrc/offline-moonshine-v2-greedy-search-decoder.h
//
// Copyright (c)  2024-2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_V2_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_V2_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-moonshine-decoder.h"
#include "sherpa-onnx/csrc/offline-moonshine-model-v2.h"

namespace sherpa_onnx {

class OfflineMoonshineV2GreedySearchDecoder : public OfflineMoonshineDecoder {
 public:
  explicit OfflineMoonshineV2GreedySearchDecoder(OfflineMoonshineModelV2 *model)
      : model_(model) {}

  std::vector<OfflineMoonshineDecoderResult> Decode(
      Ort::Value encoder_out) override;

 private:
  OfflineMoonshineModelV2 *model_;  // not owned
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_V2_GREEDY_SEARCH_DECODER_H_
