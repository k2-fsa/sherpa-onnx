// sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-whisper-decoder.h"
#include "sherpa-onnx/csrc/offline-whisper-model.h"

namespace sherpa_onnx {

class OfflineWhisperGreedySearchDecoder : public OfflineWhisperDecoder {
 public:
  OfflineWhisperGreedySearchDecoder(const OfflineWhisperModelConfig &config,
                                    OfflineWhisperModel *model)
      : config_(config), model_(model) {}

  std::vector<OfflineWhisperDecoderResult> Decode(Ort::Value cross_k,
                                                  Ort::Value cross_v) override;

  void SetConfig(const OfflineWhisperModelConfig &config) override;

 private:
  OfflineWhisperModelConfig config_;
  OfflineWhisperModel *model_;  // not owned
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_
