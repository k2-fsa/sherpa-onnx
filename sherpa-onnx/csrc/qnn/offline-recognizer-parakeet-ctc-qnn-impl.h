// sherpa-onnx/csrc/qnn/offline-recognizer-parakeet-ctc-qnn-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_PARAKEET_CTC_QNN_IMPL_H_
#define SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_PARAKEET_CTC_QNN_IMPL_H_

#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/qnn/offline-parakeet-ctc-model-qnn.h"
#include "sherpa-onnx/csrc/rknn/offline-ctc-greedy-search-decoder-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// defined in ../offline-recognizer-ctc-impl.h
OfflineRecognitionResult Convert(const OfflineCtcDecoderResult &src,
                                 const SymbolTable &sym_table,
                                 int32_t frame_shift_ms,
                                 int32_t subsampling_factor);

class OfflineRecognizerParakeetCtcQnnImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerParakeetCtcQnnImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        symbol_table_(config.model_config.tokens),
        model_(std::make_unique<OfflineParakeetCtcModelQnn>(
            config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerParakeetCtcQnnImpl(Manager *mgr,
                                      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        symbol_table_(mgr, config.model_config.tokens),
        model_(std::make_unique<OfflineParakeetCtcModelQnn>(
            mgr, config.model_config)) {
    Init();
  }

  void Init() {
    // Parakeet CTC uses the same librosa-compatible fbank features as
    // NeMo CTC (non-GigaAM). See also scripts/nemo/qnn/parakeet-ctc/test_onnx.py
    config_.feat_config.low_freq = 0;
    config_.feat_config.high_freq = 0;
    config_.feat_config.is_librosa = true;
    config_.feat_config.remove_dc_offset = false;
    config_.feat_config.window_type = "hann";
    config_.feat_config.feature_dim = model_->FeatDim();

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

  OfflineRecognizerConfig GetConfig() const override {
    return OfflineRecognizerImpl::config_;
  }

 private:
  // Decode a single stream.
  void DecodeStream(OfflineStream *s) const {
    std::vector<float> f = s->GetFrames();

    int32_t feat_dim = config_.feat_config.feature_dim;
    NemoNormalizePerFeature(f.data(), f.size() / feat_dim, feat_dim);

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
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineParakeetCtcModelQnn> model_;
  std::unique_ptr<OfflineCtcGreedySearchDecoderRknn> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_PARAKEET_CTC_QNN_IMPL_H_
