// sherpa-onnx/csrc/qnn/offline-recognizer-transducer-qnn-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_TRANSDUCER_QNN_IMPL_H_
#define SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_TRANSDUCER_QNN_IMPL_H_

#include <algorithm>
#include <fstream>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-lm.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/qnn/offline-zipformer-transducer-model-qnn.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_onnx {

namespace qnn_offline_impl {

inline std::string RemoveSpaceBetweenCjk(const std::string &text) {
  std::u32string u32 = Utf8ToUtf32(text);
  if (u32.size() < 3) {
    return text;
  }

  std::u32string ans;
  ans.reserve(u32.size());
  for (size_t i = 0; i != u32.size(); ++i) {
    if (u32[i] == U' ' && i > 0 && i + 1 < u32.size() && IsCJK(u32[i - 1]) &&
        IsCJK(u32[i + 1])) {
      continue;
    }
    ans.push_back(u32[i]);
  }

  return Utf32ToUtf8(ans);
}

inline std::vector<int32_t> GetQnnContext(const std::vector<int64_t> &tokens,
                                          int32_t context_size) {
  if (static_cast<int32_t>(tokens.size()) <= context_size) {
    return std::vector<int32_t>(tokens.begin(), tokens.end());
  }

  return std::vector<int32_t>(tokens.end() - context_size, tokens.end());
}

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

}  // namespace qnn_offline_impl

class OfflineRecognizerTransducerQnnImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerTransducerQnnImpl(const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineZipformerTransducerModelQnn>(
            config_.model_config)) {
    Init();
  }

  template <typename Manager>
  explicit OfflineRecognizerTransducerQnnImpl(Manager *mgr,
                                              const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineZipformerTransducerModelQnn>(
            mgr, config_.model_config)) {
    Init();
  }

  std::unique_ptr<OfflineStream> CreateStream(
      const std::string &hotwords) const override {
    std::string hws = hotwords;
    std::replace(hws.begin(), hws.end(), '/', '\n');
    std::istringstream is(hws);
    std::vector<std::vector<int32_t>> current;
    std::vector<float> current_scores;
    if (!EncodeHotwords(is, config_.model_config.modeling_unit, symbol_table_,
                        bpe_encoder_.get(), &current, &current_scores)) {
      SHERPA_ONNX_LOGE("Encode hotwords failed, skipping, hotwords are : '%s'",
                       hotwords.c_str());
    }

    int32_t num_default_hws = hotwords_.size();
    int32_t num_hws = current.size();
    current.insert(current.end(), hotwords_.begin(), hotwords_.end());

    if (!current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else if (!current_scores.empty() && boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_default_hws,
                            config_.hotwords_score);
    } else if (current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_hws,
                            config_.hotwords_score);
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    }

    auto context_graph = std::make_shared<ContextGraph>(
        current, config_.hotwords_score, current_scores);
    return std::make_unique<OfflineStream>(config_.feat_config, context_graph);
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config, hotwords_graph_);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto result = DecodeStream(ss[i]);
      auto r = qnn_offline_impl::Convert(result, symbol_table_, frame_shift_ms,
                                         model_->SubsamplingFactor());
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void Init() {
    if (symbol_table_.Contains("<unk>")) {
      unk_id_ = symbol_table_["<unk>"];
    }

    if (config_.decoding_method == "modified_beam_search") {
      if (!config_.lm_config.model.empty()) {
        lm_ = OfflineLM::Create(config_.lm_config);
      }

      if (!config_.model_config.bpe_vocab.empty()) {
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
            config_.model_config.bpe_vocab);
      }

      if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }
    } else if (config_.decoding_method != "greedy_search") {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  OfflineTransducerDecoderResult DecodeStream(OfflineStream *s) const {
    auto features = s->GetFrames();
    int32_t feat_dim = s->FeatureDim();
    int32_t num_frames = static_cast<int32_t>(features.size()) / feat_dim;
    auto encoder_out = model_->RunEncoder(std::move(features));

    if (config_.decoding_method == "greedy_search") {
      return GreedySearch(encoder_out);
    }

    return ModifiedBeamSearch(s, encoder_out, num_frames);
  }

  OfflineTransducerDecoderResult GreedySearch(
      const std::vector<float> &encoder_out) const {
    OfflineTransducerDecoderResult ans;

    std::vector<int64_t> tokens(model_->ContextSize(), -1);
    tokens.back() = 0;  // blank_id

    auto decoder_out =
        model_->RunDecoder(qnn_offline_impl::GetQnnContext(tokens, model_->ContextSize()));

    int32_t vocab_size = model_->VocabSize();
    int32_t encoder_dim = model_->EncoderDim();
    int32_t num_encoder_frames =
        static_cast<int32_t>(encoder_out.size()) / encoder_dim;

    for (int32_t t = 0; t != num_encoder_frames; ++t) {
      auto logit = model_->RunJoiner(encoder_out.data() + t * encoder_dim,
                                     decoder_out);
      if (blank_penalty_ > 0.f) {
        logit[0] -= blank_penalty_;
      }

      LogSoftmax(logit.data(), vocab_size);
      auto y = static_cast<int32_t>(std::distance(
          logit.begin(), std::max_element(logit.begin(), logit.end())));
      float log_prob = logit[y];

      if (y != 0 && y != unk_id_) {
        tokens.push_back(y);
        ans.timestamps.push_back(t);
        ans.ys_log_probs.push_back(log_prob);
        decoder_out = model_->RunDecoder(
            qnn_offline_impl::GetQnnContext(tokens, model_->ContextSize()));
      }
    }

    ans.tokens.assign(tokens.begin() + model_->ContextSize(), tokens.end());
    return ans;
  }

  OfflineTransducerDecoderResult ModifiedBeamSearch(
      OfflineStream *s, const std::vector<float> &encoder_out,
      int32_t num_frames) const {
    int32_t vocab_size = model_->VocabSize();
    int32_t context_size = model_->ContextSize();
    int32_t encoder_dim = model_->EncoderDim();
    int32_t num_encoder_frames =
        static_cast<int32_t>(encoder_out.size()) / encoder_dim;

    std::vector<int64_t> blanks(context_size, -1);
    blanks.back() = 0;  // blank_id

    const ContextState *context_state = nullptr;
    if (s->GetContextGraph() != nullptr) {
      context_state = s->GetContextGraph()->Root();
    }

    Hypotheses cur({Hypothesis(blanks, 0, context_state)});

    for (int32_t t = 0; t != num_encoder_frames; ++t) {
      const float *cur_encoder_out = encoder_out.data() + t * encoder_dim;
      auto prev = cur.GetTopK(config_.max_active_paths, false);

      Hypotheses next;
      for (const auto &hyp : prev) {
        auto decoder_out = model_->RunDecoder(
            qnn_offline_impl::GetQnnContext(hyp.ys, context_size));
        auto logit = model_->RunJoiner(cur_encoder_out, decoder_out);
        if (blank_penalty_ > 0.f) {
          logit[0] -= blank_penalty_;
        }
        LogSoftmax(logit.data(), vocab_size);

        Hypothesis blank_hyp = hyp;
        blank_hyp.log_prob += logit[0];
        ++blank_hyp.num_trailing_blanks;
        next.Add(std::move(blank_hyp));

        auto topk = TopkIndex(logit.data(), vocab_size, config_.max_active_paths);
        for (auto y : topk) {
          if (y == 0 || y == unk_id_) {
            continue;
          }

          Hypothesis new_hyp = hyp;
          new_hyp.ys.push_back(y);
          new_hyp.timestamps.push_back(t);
          new_hyp.ys_probs.push_back(logit[y]);
          new_hyp.num_trailing_blanks = 0;

          float context_score = 0;
          if (s->GetContextGraph() != nullptr) {
            auto context_res = s->GetContextGraph()->ForwardOneStep(
                new_hyp.context_state, y, false /* strict_mode */);
            context_score = std::get<0>(context_res);
            new_hyp.context_state = std::get<1>(context_res);
          }

          new_hyp.log_prob += logit[y] + context_score;
          next.Add(std::move(new_hyp));
        }
      }

      cur = Hypotheses(next.GetTopK(config_.max_active_paths, false));
    }

    for (auto iter = cur.begin(); iter != cur.end(); ++iter) {
      if (s->GetContextGraph() != nullptr) {
        auto context_res = s->GetContextGraph()->Finalize(iter->second.context_state);
        iter->second.log_prob += context_res.first;
        iter->second.context_state = context_res.second;
      }
    }

    if (lm_) {
      std::vector<Hypotheses> cur_vec(1, cur);
      lm_->ComputeLMScore(config_.lm_config.scale, context_size, &cur_vec);
      cur = std::move(cur_vec[0]);
    }

    Hypothesis hyp = cur.GetMostProbable(true);

    OfflineTransducerDecoderResult ans;
    ans.tokens.assign(hyp.ys.begin() + context_size, hyp.ys.end());
    ans.timestamps = std::move(hyp.timestamps);
    ans.ys_log_probs = std::move(hyp.ys_probs);
    (void)num_frames;
    return ans;
  }

  void InitHotwords() {
    std::ifstream is(config_.hotwords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: '%s'",
                       config_.hotwords_file.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!EncodeHotwords(is, config_.model_config.modeling_unit, symbol_table_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Failed to encode some hotwords, skip them already, see logs above "
          "for details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::vector<std::vector<int32_t>> hotwords_;
  std::vector<float> boost_scores_;
  ContextGraphPtr hotwords_graph_;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_encoder_;
  std::unique_ptr<OfflineZipformerTransducerModelQnn> model_;
  std::unique_ptr<OfflineLM> lm_;
  int32_t unk_id_ = -1;
  float blank_penalty_ = config_.blank_penalty;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_TRANSDUCER_QNN_IMPL_H_
