// sherpa-onnx/csrc/offline-recognizer-whisper-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-whisper-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

class OfflineRecognizerWhisperImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerWhisperImpl(const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineWhisperModel>(config.model_config)) {
    exit(-1);
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {}

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineWhisperModel> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
