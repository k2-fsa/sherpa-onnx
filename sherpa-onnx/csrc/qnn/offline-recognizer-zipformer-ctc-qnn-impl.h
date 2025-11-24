// sherpa-onnx/csrc/qnn/offline-recognizer-zipformer-ctc-qnn-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_ZIPFORMER_CTC_QNN_IMPL_H_
#define SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_ZIPFORMER_CTC_QNN_IMPL_H_

#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/qnn/offline-zipformer-ctc-model-qnn.h"
#include "sherpa-onnx/csrc/rknn/offline-ctc-greedy-search-decoder-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// defined in ../offline-recognizer-ctc-impl.h
OfflineRecognitionResult Convert(const OfflineCtcDecoderResult &src,
                                 const SymbolTable &sym_table,
                                 int32_t frame_shift_ms,
                                 int32_t subsampling_factor);

class OfflineRecognizerZipformerCtcQnnImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerZipformerCtcQnnImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineZipformerCtcModelQnn>(
            config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerZipformerCtcQnnImpl(Manager *mgr,
                                       const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineZipformerCtcModelQnn>(
            mgr, config.model_config)) {
    Init();
  }

  void Init() {
    if (config_.decoding_method == "greedy_search") {
      if (!symbol_table_.Contains("<blk>") &&
          !symbol_table_.Contains("<eps>") &&
          !symbol_table_.Contains("<blank>") &&
          config_.model_config.omnilingual.model.empty()) {
        // for omnilingual asr, its blank id is 0
        SHERPA_ONNX_LOGE(
            "We expect that tokens.txt contains "
            "the symbol <blk> or <eps> or <blank> and its ID.");
        SHERPA_ONNX_EXIT(-1);
      }

      int32_t blank_id = 0;
      if (symbol_table_.Contains("<blk>")) {
        blank_id = symbol_table_["<blk>"];
      } else if (symbol_table_.Contains("<eps>")) {
        // for tdnn models of the yesno recipe from icefall
        blank_id = symbol_table_["<eps>"];
      } else if (symbol_table_.Contains("<blank>")) {
        // for Wenet CTC models
        blank_id = symbol_table_["<blank>"];
      }

      decoder_ = std::make_unique<OfflineCtcGreedySearchDecoderRknn>(blank_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  // Decode a single stream.
  // Some models do not support batch size > 1, e.g., WeNet CTC models.
  void DecodeStream(OfflineStream *s) const {
    std::vector<float> f = s->GetFrames();

    int32_t vocab_size = model_->VocabSize();

    std::vector<float> log_probs = model_->Run(std::move(f));
    int32_t num_out_frames = log_probs.size() / vocab_size;

    auto result =
        decoder_->Decode(log_probs.data(), num_out_frames, vocab_size);

    int32_t frame_shift_ms = 10;

    auto r = Convert(result, symbol_table_, frame_shift_ms,
                     model_->SubsamplingFactor());
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    s->SetResult(r);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineZipformerCtcModelQnn> model_;
  std::unique_ptr<OfflineCtcGreedySearchDecoderRknn> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_ZIPFORMER_CTC_QNN_IMPL_H_
