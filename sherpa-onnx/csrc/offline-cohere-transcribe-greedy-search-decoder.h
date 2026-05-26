// sherpa-onnx/csrc/offline-cohere-transcribe-greedy-search-decoder.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-cohere-transcribe-decoder.h"
#include "sherpa-onnx/csrc/offline-cohere-transcribe-model.h"

namespace sherpa_onnx {

class OfflineCohereTranscribeGreedySearchDecoder
    : public OfflineCohereTranscribeDecoder {
 public:
  OfflineCohereTranscribeGreedySearchDecoder(
      OfflineCohereTranscribeModel *model)
      : model_(model) {}

  std::vector<OfflineCohereTranscribeDecoderResult> Decode(
      Ort::Value cross_k, Ort::Value cross_v,
      const std::vector<int64_t> &prompt, int32_t eos,
      int32_t num_feature_frames) override;

 private:
  OfflineCohereTranscribeModel *model_;  // not owned
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_GREEDY_SEARCH_DECODER_H_
