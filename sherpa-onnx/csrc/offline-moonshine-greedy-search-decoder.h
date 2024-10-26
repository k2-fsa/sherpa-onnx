// sherpa-onnx/csrc/offline-moonshine-greedy-search-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-moonshine-decoder.h"
#include "sherpa-onnx/csrc/offline-moonshine-model.h"

namespace sherpa_onnx {

class OfflineMoonshineGreedySearchDecoder : public OfflineMoonshineDecoder {
 public:
  explicit OfflineMoonshineGreedySearchDecoder(OfflineMoonshineModel *model)
      : model_(model) {}

  std::vector<OfflineMoonshineDecoderResult> Decode(
      Ort::Value encoder_out) override;

 private:
  OfflineMoonshineModel *model_;  // not owned
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_GREEDY_SEARCH_DECODER_H_
