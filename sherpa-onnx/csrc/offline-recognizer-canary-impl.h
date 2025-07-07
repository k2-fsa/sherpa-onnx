// sherpa-onnx/csrc/offline-recognizer-canary-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CANARY_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CANARY_IMPL_H_

#include <fstream>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-canary-model.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

class OfflineRecognizerCanaryImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerCanaryImpl(const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineCanaryModel>(config_.model_config)) {
    PostInit();
  }

  template <typename Manager>
  explicit OfflineRecognizerCanaryImpl(Manager *mgr,
                                       const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(
            std::make_unique<OfflineCanaryModel>(mgr, config_.model_config)) {
    PostInit();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  void DecodeStream(OfflineStream *s) const {}

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void PostInit() {
    auto &meta = model_->GetModelMetadata();
    config_.feat_config.feature_dim = meta.feat_dim;

    config_.feat_config.nemo_normalize_type = meta.normalize_type;

    config_.feat_config.dither = 0;
    config_.feat_config.remove_dc_offset = false;
    config_.feat_config.low_freq = 0;
    config_.feat_config.window_type = "hann";
    config_.feat_config.is_librosa = true;

    meta.lang2id["en"] = symbol_table_["<|en|>"];
    meta.lang2id["es"] = symbol_table_["<|es|>"];
    meta.lang2id["de"] = symbol_table_["<|de|>"];
    meta.lang2id["fr"] = symbol_table_["<|fr|>"];

    if (symbol_table_.NumSymbols() != meta.vocab_size) {
      SHERPA_ONNX_LOGE("number of lines in tokens.txt %d != %d (vocab_size)",
                       symbol_table_.NumSymbols(), meta.vocab_size);
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineCanaryModel> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CANARY_IMPL_H_
