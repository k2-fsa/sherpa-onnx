// sherpa-onnx/csrc/offline-recognizer-ctc-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_

#include <memory>

#include "sherpa-onnx/csrc/offline-ctc-model.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

class OfflineRecognizerCtcImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerCtcImpl(const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(OfflineCtcModel::Create(config_.model_config)) {}

  std::unique_ptr<OfflineStream> CreateStream() const override {
    SHERPA_ONNX_LOGE("create stream");
    // return std::make_unique<OfflineStream>(config_.feat_config);
    return nullptr;
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    SHERPA_ONNX_LOGE("decoding");
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineCtcModel> model_;
  // std::unique_ptr<OfflineTransducerDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_
