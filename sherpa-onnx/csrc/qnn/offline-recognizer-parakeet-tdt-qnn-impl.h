// sherpa-onnx/csrc/qnn/offline-recognizer-parakeet-tdt-qnn-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_PARAKEET_TDT_QNN_IMPL_H_
#define SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_PARAKEET_TDT_QNN_IMPL_H_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/qnn/offline-parakeet-tdt-model-qnn.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

namespace qnn_parakeet_tdt_impl {

inline OfflineRecognitionResult Convert(
    const OfflineTransducerDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());
  r.durations.reserve(src.durations.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    text.append(sym);

    if (sym.size() == 1 && (sym[0] < 0x20 || sym[0] > 0x7e)) {
      std::ostringstream os;
      os << "<0x" << std::hex << std::uppercase
         << (static_cast<int32_t>(sym[0]) & 0xff) << ">";
      sym = os.str();
    }

    r.tokens.push_back(std::move(sym));
  }

  if (sym_table.IsByteBpe()) {
    text = sym_table.DecodeByteBpe(text);
  }

  text = RemoveSpaceBetweenCjk(text);

  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000.f * subsampling_factor;
  for (auto t : src.timestamps) {
    r.timestamps.push_back(frame_shift_s * t);
  }

  for (auto d : src.durations) {
    r.durations.push_back(d * frame_shift_s);
  }

  r.ys_log_probs = src.ys_log_probs;
  return r;
}

}  // namespace qnn_parakeet_tdt_impl

class OfflineRecognizerParakeetTdtQnnImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerParakeetTdtQnnImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineParakeetTdtModelQnn>(
            config_.model_config)) {
    Init();
  }

  template <typename Manager>
  explicit OfflineRecognizerParakeetTdtQnnImpl(
      Manager *mgr, const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineParakeetTdtModelQnn>(
            mgr, config_.model_config)) {
    Init();
  }

  std::unique_ptr<OfflineStream> CreateStream(
      const std::string & /*hotwords*/) const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto result = DecodeStream(ss[i]);
      auto r = qnn_parakeet_tdt_impl::Convert(
          result, symbol_table_, frame_shift_ms, model_->SubsamplingFactor());

      // Remove leading space from BPE tokenization
      if (!r.text.empty() && r.text.front() == ' ') {
        r.text.erase(0, 1);
      }

      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void Init() {
    if (!symbol_table_.Contains("<blk>")) {
      SHERPA_ONNX_LOGE("tokens.txt does not include the blank token <blk>");
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t blank_id = symbol_table_["<blk>"];
    int32_t vocab_size = blank_id + 1;

    if (symbol_table_.NumSymbols() != vocab_size) {
      SHERPA_ONNX_LOGE("number of lines in tokens.txt %d != %d (vocab_size)",
                       symbol_table_.NumSymbols(), vocab_size);
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.decoding_method != "greedy_search") {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported for parakeet TDT QNN. "
          "Given: %s",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    // Setup feat_config similar to offline-recognizer-transducer-nemo-impl.h
    int32_t feat_dim = model_->FeatDim();
    if (feat_dim > 0) {
      config_.feat_config.feature_dim = feat_dim;
    }

    config_.feat_config.nemo_normalize_type = "per_feature";
    config_.feat_config.low_freq = 0;
    config_.feat_config.is_librosa = true;
    config_.feat_config.remove_dc_offset = false;

    if (config_.model_config.debug) {
      SHERPA_ONNX_LOGE("blank_id: %d", blank_id);
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size);
      SHERPA_ONNX_LOGE("subsampling_factor: %d", model_->SubsamplingFactor());
      SHERPA_ONNX_LOGE("encoder_dim: %d", model_->EncoderDim());
      SHERPA_ONNX_LOGE("feat_dim: %d", feat_dim);
    }
  }

  OfflineTransducerDecoderResult DecodeStream(OfflineStream *s) const {
    auto features = s->GetFrames();
    auto encoder_out = model_->RunEncoder(std::move(features));
    return GreedySearch(encoder_out);
  }

  OfflineTransducerDecoderResult GreedySearch(
      const std::vector<float> &encoder_out) const {
    OfflineTransducerDecoderResult ans;

    int32_t blank_id = symbol_table_["<blk>"];
    int32_t vocab_size = symbol_table_.NumSymbols();
    int32_t encoder_dim = model_->EncoderDim();
    int32_t num_encoder_frames =
        static_cast<int32_t>(encoder_out.size()) / encoder_dim;

    // Initialize decoder with blank token and zero states
    auto states = model_->GetDecoderInitStates();
    auto [decoder_out, next_states] =
        model_->RunDecoder(blank_id, std::move(states));
    states = std::move(next_states);

    // TDT decoding loop
    int32_t max_tokens_per_frame = 5;
    int32_t tokens_this_frame = 0;
    int32_t skip = 0;

    for (int32_t t = 0; t < num_encoder_frames; t += skip) {
      // The joiner output is named "log_probs" but actually contains raw
      // logits (log_softmax is NOT applied by the joiner).
      auto logits =
          model_->RunJoiner(encoder_out.data() + t * encoder_dim, decoder_out);

      // Split into token logits and duration logits
      float *token_logits = logits.data();
      int32_t output_size = static_cast<int32_t>(logits.size());
      int32_t num_durations = output_size - vocab_size;
      const float *duration_logits = logits.data() + vocab_size;

      auto y = MaxElementIndex(token_logits, vocab_size);

      // Duration prediction
      if (num_durations > 0) {
        skip = MaxElementIndex(duration_logits, num_durations);
      } else {
        skip = 0;
      }

      if (y != blank_id) {
        LogSoftmax(token_logits, vocab_size);
        float log_prob = token_logits[y];

        ans.tokens.push_back(y);
        ans.timestamps.push_back(t);
        ans.durations.push_back(skip);
        ans.ys_log_probs.push_back(log_prob);

        auto [new_decoder_out, new_states] =
            model_->RunDecoder(y, std::move(states));
        decoder_out = std::move(new_decoder_out);
        states = std::move(new_states);

        tokens_this_frame += 1;
      }

      if (skip > 0) {
        tokens_this_frame = 0;
      }

      if (tokens_this_frame >= max_tokens_per_frame) {
        tokens_this_frame = 0;
        skip = 1;
      }

      if (y == blank_id && skip == 0) {
        tokens_this_frame = 0;
        skip = 1;
      }
    }

    return ans;
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineParakeetTdtModelQnn> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_PARAKEET_TDT_QNN_IMPL_H_
