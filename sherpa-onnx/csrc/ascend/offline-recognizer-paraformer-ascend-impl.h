// sherpa-onnx/csrc/ascend/offline-recognizer-paraformer-ascend-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ASCEND_OFFLINE_RECOGNIZER_PARAFORMER_ASCEND_IMPL_H_
#define SHERPA_ONNX_CSRC_ASCEND_OFFLINE_RECOGNIZER_PARAFORMER_ASCEND_IMPL_H_

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/ascend/offline-paraformer-model-ascend.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/rknn/offline-ctc-greedy-search-decoder-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// defined in ../online-recognizer-paraformer-impl.h
OfflineRecognitionResult Convert(const OfflineParaformerDecoderResult &src,
                                 const SymbolTable &sym_table);

class OfflineRecognizerParaformerAscendImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerParaformerAscendImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineParaformerModelAscend>(
            config.model_config)) {
    if (config.decoding_method != "greedy_search") {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    InitFeatConfig();
  }

  template <typename Manager>
  OfflineRecognizerParaformerAscendImpl(Manager *mgr,
                                        const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineParaformerModelAscend>(
            mgr, config.model_config)) {
    if (config.decoding_method != "greedy_search") {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    InitFeatConfig();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeOneStream(ss[i]);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void InitFeatConfig() {
    config_.feat_config.normalize_samples = false;
    config_.feat_config.window_type = "hamming";
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
  }

  void DecodeOneStream(OfflineStream *s) const {
    std::vector<float> f = s->GetFrames();

    std::vector<float> logits = model_->Run(std::move(f));
    if (logits.empty()) {
      SHERPA_ONNX_LOGE("No speech detected");
      return;
    }

    int32_t vocab_size = model_->VocabSize();
    int32_t num_tokens = logits.size() / vocab_size;

    int32_t eos_id = symbol_table_["</s>"];

    OfflineParaformerDecoderResult r;
    const float *p = logits.data();
    for (int32_t i = 0; i < num_tokens; ++i) {
      auto max_idx = static_cast<int64_t>(
          std::distance(p, std::max_element(p, p + vocab_size)));

      if (max_idx == eos_id) {
        break;
      }
      r.tokens.push_back(max_idx);
      p += vocab_size;
    }

    auto result = Convert(r, symbol_table_);
    result.text = ApplyInverseTextNormalization(std::move(result.text));
    result.text = ApplyHomophoneReplacer(std::move(result.text));
    s->SetResult(result);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineParaformerModelAscend> model_;
  std::unique_ptr<OfflineCtcGreedySearchDecoderRknn> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_OFFLINE_RECOGNIZER_PARAFORMER_ASCEND_IMPL_H_
