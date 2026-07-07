// sherpa-onnx/csrc/qnn/online-recognizer-nemo-transducer-qnn-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_ONLINE_RECOGNIZER_NEMO_TRANSDUCER_QNN_IMPL_H_
#define SHERPA_ONNX_CSRC_QNN_ONLINE_RECOGNIZER_NEMO_TRANSDUCER_QNN_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/endpoint.h"
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/qnn/online-nemo-transducer-model-qnn.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace nemo_transducer_qnn_impl {

inline OnlineRecognizerResult ConvertResult(
    OnlineTransducerDecoderResultNoOrt src, const SymbolTable &sym_table,
    float frame_shift_ms, int32_t subsampling_factor, int32_t segment,
    int32_t frames_since_start) {
  OnlineRecognizerResult r;

  // The first token is the initial blank_id_ used to bootstrap the decoder.
  // Remove it before processing.
  if (!src.tokens.empty()) {
    src.tokens.erase(src.tokens.begin());
  }

  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());
  r.ys_probs = std::move(src.ys_probs);

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

  r.segment = segment;
  r.start_time = frames_since_start * frame_shift_ms / 1000.f;

  return r;
}

}  // namespace nemo_transducer_qnn_impl

class OnlineRecognizerNemoTransducerQnnImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerNemoTransducerQnnImpl(
      const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        endpoint_(config.endpoint_config),
        model_(std::make_unique<OnlineNemoTransducerModelQnn>(
            config.model_config)),
        blank_id_(model_->VocabSize() - 1) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      sym_ = SymbolTable(config.model_config.tokens, true);
    }
    ValidateBlankId();
    SetupFeatConfig();
    InitResultTemplate();
  }

  template <typename Manager>
  OnlineRecognizerNemoTransducerQnnImpl(Manager *mgr,
                                     const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        endpoint_(config.endpoint_config),
        model_(std::make_unique<OnlineNemoTransducerModelQnn>(
            mgr, config.model_config)),
        blank_id_(model_->VocabSize() - 1) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      sym_ = SymbolTable(mgr, config.model_config.tokens);
    }
    ValidateBlankId();
    SetupFeatConfig();
    InitResultTemplate();
  }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto s = std::make_unique<OnlineStream>(config_.feat_config);
    s->SetQnnStates(model_->GetEncoderInitStates());
    s->SetQnnResult(result_template_);
    return s;
  }

  std::unique_ptr<OnlineStream> CreateStream(
      const std::string &hotwords) const override {
    if (!hotwords.empty()) {
      SHERPA_ONNX_LOGE("hotwords are not supported with qnn nemo transducer");
      SHERPA_ONNX_EXIT(-1);
    }
    return CreateStream();
  }

  bool IsReady(OnlineStream *s) const override {
    return s->NumFramesReady() - s->GetNumProcessedFrames() >=
           model_->WindowSize();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      auto *s = ss[i];
      std::vector<float> features =
          s->GetFrames(s->GetNumProcessedFrames(), model_->WindowSize());

      auto encoder_out = model_->RunEncoder(
          std::move(features), model_->WindowSize(), &s->GetQnnStates());

      int32_t num_processed = s->GetNumProcessedFrames();
      // Each encoder output frame corresponds to subsampling_factor input
      // frames. The frame_offset is in terms of encoder output frames.
      int32_t frame_offset = num_processed / subsampling_factor_;

      int32_t encoder_dim = model_->EncoderDim();
      int32_t num_encoder_frames =
          static_cast<int32_t>(encoder_out.size()) / encoder_dim;
      const float *p = encoder_out.data();

      // TDT-style greedy decoding loop.
      // For each encoder output frame, try up to max_symbols_per_frame
      // non-blank predictions before moving to the next frame.
      int32_t max_symbols_per_frame = 10;
      int32_t num_symbols = 0;

      for (int32_t t = 0; t < num_encoder_frames;) {
        auto &r = s->GetQnnResult();
        int32_t frame_index = frame_offset + t;

        auto logit =
            model_->RunJoiner(p + t * encoder_dim, r.decoder_out);
        auto y = MaxElementIndex(logit.data(), logit.size());

        if (y != blank_id_) {
          r.tokens.push_back(y);
          r.timestamps.push_back(frame_index);
          r.num_trailing_blanks = 0;

          // Run decoder with the new token and current states.
          auto [decoder_out, next_states] =
              model_->RunDecoder(y, std::move(r.states));
          r.decoder_out = std::move(decoder_out);
          r.states = std::move(next_states);

          ++num_symbols;
          if (num_symbols > max_symbols_per_frame) {
            num_symbols = 0;
            ++t;
          }
        } else {
          ++r.num_trailing_blanks;
          ++t;
          num_symbols = 0;
        }
      }

      s->GetNumProcessedFrames() += model_->WindowShift();
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    int32_t frame_shift_ms = 10;
    auto r = nemo_transducer_qnn_impl::ConvertResult(
        s->GetQnnResult(), sym_, frame_shift_ms, subsampling_factor_,
        s->GetCurrentSegment(), s->GetNumFramesSinceStart());
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    return r;
  }

  bool IsEndpoint(OnlineStream *s) const override {
    if (!config_.enable_endpoint) {
      return false;
    }

    int32_t num_processed_frames = s->GetNumProcessedFrames();
    float frame_shift_in_seconds = 0.01f;
    int32_t trailing_silence_frames =
        s->GetQnnResult().num_trailing_blanks * subsampling_factor_;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    const auto &last = s->GetQnnResult();
    if (!last.tokens.empty() && last.tokens.back() != blank_id_ &&
        static_cast<int32_t>(last.tokens.size()) > 1) {
      s->GetCurrentSegment() += 1;
    }

    OnlineTransducerDecoderResultNoOrt r;
    r.tokens.push_back(blank_id_);

    // Carry over decoder states from the last result for warm-up.
    if (!last.states.empty()) {
      auto [decoder_out, next_states] =
          model_->RunDecoder(blank_id_, last.states);
      r.decoder_out = std::move(decoder_out);
      r.states = std::move(next_states);
    } else {
      // First time: run decoder from zero states.
      auto [decoder_out, next_states] =
          model_->RunDecoder(blank_id_, model_->GetDecoderInitState());
      r.decoder_out = std::move(decoder_out);
      r.states = std::move(next_states);
    }

    if (config_.reset_encoder) {
      s->SetQnnStates(model_->GetEncoderInitStates());
    }

    s->Reset();
    s->SetQnnResult(r);
  }

 private:
  void ValidateBlankId() const {
    if (sym_.Contains(blank_id_)) {
      auto sym = sym_[blank_id_];
      if (sym != "<blk>") {
        SHERPA_ONNX_LOGE(
            "Expected blank token '<blk>' at id %d, but got '%s'",
            blank_id_, sym.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }
    if (blank_id_ != model_->VocabSize() - 1) {
      SHERPA_ONNX_LOGE(
          "blank_id %d should equal vocab_size - 1 (%d)", blank_id_,
          model_->VocabSize() - 1);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void SetupFeatConfig() {
    config_.feat_config.feature_dim = model_->FeatureDim();
    config_.feat_config.nemo_normalize_type = model_->NormalizationType();
    config_.feat_config.low_freq = 0;
    config_.feat_config.is_librosa = true;
    config_.feat_config.remove_dc_offset = false;
  }

  void InitResultTemplate() {
    result_template_.tokens.push_back(blank_id_);

    // Run decoder with blank to get the initial decoder_out and states.
    auto [decoder_out, next_states] =
        model_->RunDecoder(blank_id_, model_->GetDecoderInitState());
    result_template_.decoder_out = std::move(decoder_out);
    result_template_.states = std::move(next_states);

    subsampling_factor_ = model_->SubsamplingFactor();
  }

 private:
  OnlineRecognizerConfig config_;
  Endpoint endpoint_;
  std::unique_ptr<OnlineNemoTransducerModelQnn> model_;
  SymbolTable sym_;

  int32_t blank_id_ = 0;
  int32_t subsampling_factor_ = 8;  // will be overwritten in InitResultTemplate
  OnlineTransducerDecoderResultNoOrt result_template_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_ONLINE_RECOGNIZER_NEMO_TRANSDUCER_QNN_IMPL_H_
