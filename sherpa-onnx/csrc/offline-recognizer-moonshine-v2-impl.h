// sherpa-onnx/csrc/offline-recognizer-moonshine-v2-impl.h
//
// Copyright (c)  2024-2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_MOONSHINE_V2_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_MOONSHINE_V2_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-moonshine-decoder.h"
#include "sherpa-onnx/csrc/offline-moonshine-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-moonshine-model-v2.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

static OfflineRecognitionResult ConvertV2(
    const OfflineMoonshineDecoderResult &src, const SymbolTable &sym_table) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());

  std::string text;
  for (auto i : src.tokens) {
    if (!sym_table.Contains(i)) {
      continue;
    }

    const auto &s = sym_table[i];
    text += s;
    r.tokens.push_back(s);
  }

  r.text = text;

  return r;
}

class OfflineRecognizerMoonshineV2Impl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerMoonshineV2Impl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineMoonshineModelV2>(config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerMoonshineV2Impl(Manager *mgr,
                                   const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineMoonshineModelV2>(mgr,
                                                         config.model_config)) {
    Init();
  }

  void Init() {
    if (config_.decoding_method == "greedy_search") {
      // decoder_ =
      //     std::make_unique<OfflineMoonshineGreedySearchDecoder>(model_.get());
    } else {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported at present for moonshine. Given %s",
          config_.decoding_method.c_str());
      exit(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    MoonshineTag tag;
    return std::make_unique<OfflineStream>(tag);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    // batch decoding is not implemented yet
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeStream(OfflineStream *s) const {}

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineMoonshineModelV2> model_;
  std::unique_ptr<OfflineMoonshineDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_MOONSHINE_V2_IMPL_H_
