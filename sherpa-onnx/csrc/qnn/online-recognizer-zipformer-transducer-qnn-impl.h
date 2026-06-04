// sherpa-onnx/csrc/qnn/online-recognizer-zipformer-transducer-qnn-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_ONLINE_RECOGNIZER_ZIPFORMER_TRANSDUCER_QNN_IMPL_H_
#define SHERPA_ONNX_CSRC_QNN_ONLINE_RECOGNIZER_ZIPFORMER_TRANSDUCER_QNN_IMPL_H_

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
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/qnn/online-zipformer-transducer-model-qnn.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace qnn_impl {

inline std::vector<int32_t> GetQnnContext(const std::vector<int32_t> &tokens,
                                          int32_t context_size) {
  if (static_cast<int32_t>(tokens.size()) <= context_size) {
    return tokens;
  }

  return std::vector<int32_t>(tokens.end() - context_size, tokens.end());
}

inline std::vector<int64_t> ToInt64Tokens(const std::vector<int32_t> &tokens) {
  return std::vector<int64_t>(tokens.begin(), tokens.end());
}

inline std::vector<int32_t> ToInt32Tokens(const std::vector<int64_t> &tokens) {
  return std::vector<int32_t>(tokens.begin(), tokens.end());
}

inline std::vector<int32_t> GetQnnContext(const std::vector<int64_t> &tokens,
                                          int32_t context_size) {
  return GetQnnContext(ToInt32Tokens(tokens), context_size);
}

inline OnlineRecognizerResult ConvertQnnResult(
    OnlineTransducerDecoderResultNoOrt src, const SymbolTable &sym_table,
    float frame_shift_ms, int32_t subsampling_factor, int32_t segment,
    int32_t frames_since_start, int32_t context_size) {
  OnlineRecognizerResult r;

  if (static_cast<int32_t>(src.tokens.size()) >= context_size) {
    src.tokens.erase(src.tokens.begin(), src.tokens.begin() + context_size);
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

class QnnGreedySearchDecoder {
 public:
  explicit QnnGreedySearchDecoder(int32_t blank_id) : blank_id_(blank_id) {}

  OnlineTransducerDecoderResultNoOrt GetEmptyResult(
      const OnlineZipformerTransducerModelQnn &model) const {
    OnlineTransducerDecoderResultNoOrt r;
    r.tokens.resize(model.ContextSize(), -1);
    r.tokens.back() = blank_id_;
    return r;
  }

  void Decode(const float *encoder_out, int32_t frame_index,
              const OnlineZipformerTransducerModelQnn &model,
              OnlineTransducerDecoderResultNoOrt *r) const {
    auto logit = model.RunJoiner(encoder_out, r->decoder_out);
    auto y = static_cast<int32_t>(
        std::distance(logit.begin(), std::max_element(logit.begin(), logit.end())));

    if (y != blank_id_) {
      r->tokens.push_back(y);
      r->timestamps.push_back(frame_index);
      r->num_trailing_blanks = 0;
      auto context = GetQnnContext(r->tokens, model.ContextSize());
      r->decoder_out = model.RunDecoder(context);
    } else {
      ++r->num_trailing_blanks;
    }
  }

 private:
  int32_t blank_id_;
};

class QnnModifiedBeamSearchDecoder {
 public:
  QnnModifiedBeamSearchDecoder(int32_t blank_id, int32_t vocab_size,
                               int32_t beam)
      : blank_id_(blank_id), vocab_size_(vocab_size), beam_(beam) {}

  OnlineTransducerDecoderResultNoOrt GetEmptyResult(
      const OnlineZipformerTransducerModelQnn &model) const {
    OnlineTransducerDecoderResultNoOrt r;
    std::vector<int32_t> context(model.ContextSize(), -1);
    context.back() = blank_id_;
    r.tokens = context;
    r.hyps = Hypotheses({Hypothesis(ToInt64Tokens(context), 0)});
    return r;
  }

  void Decode(const float *encoder_out, int32_t frame_index,
              const OnlineZipformerTransducerModelQnn &model,
              OnlineTransducerDecoderResultNoOrt *r) const {
    Hypotheses next;
    auto cur = r->hyps.GetTopK(beam_, false);

    for (const auto &hyp : cur) {
      auto context = GetQnnContext(hyp.ys, model.ContextSize());
      auto decoder_out = model.RunDecoder(context);
      auto logit = model.RunJoiner(encoder_out, decoder_out);
      auto log_probs = LogSoftmax(logit);

      Hypothesis blank_hyp = hyp;
      blank_hyp.log_prob += log_probs[blank_id_];
      ++blank_hyp.num_trailing_blanks;
      next.Add(std::move(blank_hyp));

      std::vector<int32_t> indexes(vocab_size_);
      std::iota(indexes.begin(), indexes.end(), 0);
      std::partial_sort(
          indexes.begin(), indexes.begin() + std::min(beam_, vocab_size_),
          indexes.end(), [&log_probs](int32_t a, int32_t b) {
            return log_probs[a] > log_probs[b];
          });

      int32_t num = std::min(beam_, vocab_size_);
      for (int32_t i = 0; i != num; ++i) {
        int32_t y = indexes[i];
        if (y == blank_id_) {
          continue;
        }
        auto ys = hyp.ys;
        auto timestamps = hyp.timestamps;
        ys.push_back(y);
        timestamps.push_back(frame_index);
        Hypothesis new_hyp(std::move(ys), hyp.log_prob + log_probs[y]);
        new_hyp.timestamps = std::move(timestamps);
        new_hyp.num_trailing_blanks = 0;
        next.Add(std::move(new_hyp));
      }
    }

    r->hyps = Hypotheses(next.GetTopK(beam_, false));
    const auto &best = r->hyps.GetMostProbable(false);
    r->tokens = ToInt32Tokens(best.ys);
    r->timestamps = best.timestamps;
    r->num_trailing_blanks = best.num_trailing_blanks;
    r->decoder_out = model.RunDecoder(
        GetQnnContext(r->tokens, model.ContextSize()));
  }

 private:
  static std::vector<float> LogSoftmax(const std::vector<float> &x) {
    float max_value = *std::max_element(x.begin(), x.end());
    float sum = 0;
    for (auto v : x) {
      sum += std::exp(v - max_value);
    }
    float log_sum = std::log(sum) + max_value;

    std::vector<float> ans(x.size());
    for (size_t i = 0; i != x.size(); ++i) {
      ans[i] = x[i] - log_sum;
    }
    return ans;
  }

  int32_t blank_id_;
  int32_t vocab_size_;
  int32_t beam_;
};

}  // namespace qnn_impl

class OnlineRecognizerZipformerTransducerQnnImpl
    : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerZipformerTransducerQnnImpl(
      const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        endpoint_(config.endpoint_config),
        model_(std::make_unique<OnlineZipformerTransducerModelQnn>(
            config.model_config, config.feat_config.feature_dim)),
        blank_id_(0),
        greedy_decoder_(blank_id_),
        modified_beam_search_decoder_(
            blank_id_, model_->VocabSize(), std::max(1, config.max_active_paths)) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      sym_ = SymbolTable(config.model_config.tokens, true);
    }
    InitResultTemplate();
  }

  template <typename Manager>
  OnlineRecognizerZipformerTransducerQnnImpl(
      Manager *mgr, const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        endpoint_(config.endpoint_config),
        model_(
            std::make_unique<OnlineZipformerTransducerModelQnn>(
                mgr, config.model_config, config.feat_config.feature_dim)),
        blank_id_(0),
        greedy_decoder_(blank_id_),
        modified_beam_search_decoder_(
            blank_id_, model_->VocabSize(), std::max(1, config.max_active_paths)) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      sym_ = SymbolTable(mgr, config.model_config.tokens);
    }
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
      SHERPA_ONNX_LOGE("hotwords are not supported with qnn transducers");
      SHERPA_ONNX_EXIT(-1);
    }
    return CreateStream();
  }

  bool IsReady(OnlineStream *s) const override {
    return s->NumFramesReady() - s->GetNumProcessedFrames() >=
           model_->ChunkSize();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      auto *s = ss[i];
      EnsureInitialized(s);

      std::vector<float> features =
          s->GetFrames(s->GetNumProcessedFrames(), model_->ChunkSize());

      auto encoder_out = model_->RunEncoder(
          std::move(features), model_->ChunkSize(), &s->GetQnnStates());

      int32_t num_processed = s->GetNumProcessedFrames();
      int32_t frame_offset = num_processed / 4;

      int32_t encoder_dim = model_->EncoderDim();
      int32_t num_encoder_frames =
          static_cast<int32_t>(encoder_out.size()) / encoder_dim;
      const float *p = encoder_out.data();

      for (int32_t t = 0; t != num_encoder_frames; ++t) {
        auto &r = s->GetQnnResult();
        int32_t frame_index = frame_offset + t;
        if (config_.decoding_method == "greedy_search") {
          greedy_decoder_.Decode(p + t * encoder_dim, frame_index, *model_,
                                 &r);
        } else {
          modified_beam_search_decoder_.Decode(
              p + t * encoder_dim, frame_index, *model_, &r);
        }
      }

      s->GetNumProcessedFrames() += model_->ChunkShift();
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    EnsureInitialized(s);
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    auto r = qnn_impl::ConvertQnnResult(
        s->GetQnnResult(), sym_, frame_shift_ms, subsampling_factor,
        s->GetCurrentSegment(), s->GetNumFramesSinceStart(),
        model_->ContextSize());
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    return r;
  }

  bool IsEndpoint(OnlineStream *s) const override {
    EnsureInitialized(s);

    if (!config_.enable_endpoint) {
      return false;
    }

    int32_t num_processed_frames = s->GetNumProcessedFrames();
    float frame_shift_in_seconds = 0.01f;
    int32_t trailing_silence_frames = s->GetQnnResult().num_trailing_blanks * 4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    EnsureInitialized(s);

    int32_t context_size = model_->ContextSize();

    const auto &last = s->GetQnnResult();
    if (!last.tokens.empty() && last.tokens.back() != 0 &&
        static_cast<int32_t>(last.tokens.size()) > context_size) {
      s->GetCurrentSegment() += 1;
    }

    OnlineTransducerDecoderResultNoOrt r = GetEmptyResult();

    if (static_cast<int32_t>(last.tokens.size()) > context_size) {
      r.tokens = qnn_impl::GetQnnContext(last.tokens, context_size);
      r.decoder_out = model_->RunDecoder(r.tokens);

      if (config_.decoding_method == "modified_beam_search") {
        r.hyps = Hypotheses({Hypothesis(qnn_impl::ToInt64Tokens(r.tokens), 0)});
      }
    }

    if (config_.reset_encoder) {
      s->SetQnnStates(model_->GetEncoderInitStates());
    }

    s->Reset();
    s->SetQnnResult(r);
  }

 private:
  void EnsureInitialized(OnlineStream *s) const {
    if (!s->GetQnnStates().empty()) {
      return;
    }

    s->SetQnnStates(model_->GetEncoderInitStates());
    s->SetQnnResult(result_template_);
  }

  void InitResultTemplate() {
    result_template_ = GetEmptyResult();
  }

  OnlineTransducerDecoderResultNoOrt GetEmptyResult() const {
    OnlineTransducerDecoderResultNoOrt r;
    if (config_.decoding_method == "greedy_search") {
      r = greedy_decoder_.GetEmptyResult(*model_);
    } else if (config_.decoding_method == "modified_beam_search") {
      r = modified_beam_search_decoder_.GetEmptyResult(*model_);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method for qnn transducer: %s",
                       config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    r.decoder_out =
        model_->RunDecoder(qnn_impl::GetQnnContext(r.tokens, model_->ContextSize()));
    return r;
  }

 private:
  OnlineRecognizerConfig config_;
  Endpoint endpoint_;
  std::unique_ptr<OnlineZipformerTransducerModelQnn> model_;
  SymbolTable sym_;

  int32_t blank_id_ = 0;
  qnn_impl::QnnGreedySearchDecoder greedy_decoder_;
  qnn_impl::QnnModifiedBeamSearchDecoder modified_beam_search_decoder_;
  OnlineTransducerDecoderResultNoOrt result_template_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_ONLINE_RECOGNIZER_ZIPFORMER_TRANSDUCER_QNN_IMPL_H_
