// sherpa-onnx/csrc/offline-recognizer-transducer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_

#include <algorithm>
#include <fstream>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-model.h"
#include "sherpa-onnx/csrc/offline-transducer-modified-beam-search-decoder.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_onnx {

// Converts token IDs to their symbols, appending the detokenized text to
// *text. Shared by the 1-best result and every entry of the n-best list.
static std::vector<std::string> ConvertTokens(
    const std::vector<int64_t> &token_ids, const SymbolTable &sym_table,
    std::string *text) {
  std::vector<std::string> tokens;
  tokens.reserve(token_ids.size());

  std::string t;
  for (auto i : token_ids) {
    auto sym = sym_table[i];
    t.append(sym);

    if (sym.size() == 1 && (sym[0] < 0x20 || sym[0] > 0x7e)) {
      // for bpe models with byte_fallback,
      // (but don't rewrite printable characters 0x20..0x7e,
      //  which collide with standard BPE units)
      std::ostringstream os;
      os << "<0x" << std::hex << std::uppercase
         << (static_cast<int32_t>(sym[0]) & 0xff) << ">";
      sym = os.str();
    }

    tokens.push_back(std::move(sym));
  }

  if (sym_table.IsByteBpe()) {
    t = sym_table.DecodeByteBpe(t);
  }

  *text = RemoveSpaceBetweenCjk(t);

  return tokens;
}

static std::vector<float> ConvertTimestamps(
    const std::vector<int32_t> &timestamps, float frame_shift_s) {
  std::vector<float> ans;
  ans.reserve(timestamps.size());
  for (auto t : timestamps) {
    ans.push_back(frame_shift_s * t);
  }
  return ans;
}

static OfflineRecognitionResult Convert(
    const OfflineTransducerDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.durations.reserve(src.durations.size());

  r.tokens = ConvertTokens(src.tokens, sym_table, &r.text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  r.timestamps = ConvertTimestamps(src.timestamps, frame_shift_s);

  // Copy durations (if present)
  for (auto d : src.durations) {
    r.durations.push_back(d * frame_shift_s);
  }

  // Copy token log probabilities (confidence scores)
  r.ys_log_probs = src.ys_log_probs;

  // n-best list, present only for modified_beam_search with
  // num_return_paths > 1
  r.hypotheses.reserve(src.hypotheses.size());
  for (const auto &h : src.hypotheses) {
    OfflineRecognitionHypothesis hyp;
    hyp.tokens = ConvertTokens(h.tokens, sym_table, &hyp.text);
    hyp.timestamps = ConvertTimestamps(h.timestamps, frame_shift_s);
    hyp.ys_log_probs = h.ys_log_probs;
    hyp.score = h.score;
    r.hypotheses.push_back(std::move(hyp));
  }

  return r;
}

class OfflineRecognizerTransducerImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerTransducerImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerModel>(config_.model_config)) {
    if (symbol_table_.Contains("<unk>")) {
      unk_id_ = symbol_table_["<unk>"];
    }

    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineTransducerGreedySearchDecoder>(
          model_.get(), unk_id_, config_.blank_penalty);
    } else if (config_.decoding_method == "modified_beam_search") {
      if (!config_.lm_config.model.empty()) {
        lm_ = OfflineLM::Create(config.lm_config);
      }

      if (!config_.model_config.bpe_vocab.empty()) {
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
            config_.model_config.bpe_vocab);
      }

      if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }

      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale, unk_id_, config_.blank_penalty,
          std::min(config_.num_return_paths, config_.max_active_paths));
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  template <typename Manager>
  explicit OfflineRecognizerTransducerImpl(
      Manager *mgr, const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerModel>(mgr,
                                                        config_.model_config)) {
    if (symbol_table_.Contains("<unk>")) {
      unk_id_ = symbol_table_["<unk>"];
    }

    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineTransducerGreedySearchDecoder>(
          model_.get(), unk_id_, config_.blank_penalty);
    } else if (config_.decoding_method == "modified_beam_search") {
      if (!config_.lm_config.model.empty()) {
        lm_ = OfflineLM::Create(mgr, config.lm_config);
      }

      if (!config_.model_config.bpe_vocab.empty()) {
        auto buf = ReadFile(mgr, config_.model_config.bpe_vocab);
        std::istringstream iss(std::string(buf.begin(), buf.end()));
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(iss);
      }

      if (!config_.hotwords_file.empty()) {
        InitHotwords(mgr);
      }

      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale, unk_id_, config_.blank_penalty,
          std::min(config_.num_return_paths, config_.max_active_paths));
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream(
      const std::string &hotwords) const override {
    auto hws = std::regex_replace(hotwords, std::regex("/"), "\n");
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
    } else {
      // Do nothing.
    }

    auto context_graph = std::make_shared<ContextGraph>(
        current, config_.hotwords_score, current_scores);
    return std::make_unique<OfflineStream>(config_.feat_config, context_graph);
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config,
                                           hotwords_graph_);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = ss[0]->FeatureDim();

    std::vector<Ort::Value> features;

    features.reserve(n);

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int64_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      auto f = ss[i]->GetFrames();
      int32_t num_frames = f.size() / feat_dim;

      features_length_vec[i] = num_frames;
      features_vec[i] = std::move(f);

      std::array<int64_t, 2> shape = {num_frames, feat_dim};

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }

    std::vector<const Ort::Value *> features_pointer(n);
    for (int32_t i = 0; i != n; ++i) {
      features_pointer[i] = &features[i];
    }

    std::array<int64_t, 1> features_length_shape = {n};
    Ort::Value x_length = Ort::Value::CreateTensor(
        memory_info, features_length_vec.data(), n,
        features_length_shape.data(), features_length_shape.size());

    Ort::Value x = PadSequence(model_->Allocator(), features_pointer,
                               -23.025850929940457f);

    auto t = model_->RunEncoder(std::move(x), std::move(x_length));
    auto results =
        decoder_->Decode(std::move(t.first), std::move(t.second), ss, n);

    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_, frame_shift_ms,
                       model_->SubsamplingFactor());
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));

      // Apply the same text post-processing to every hypothesis, so that
      // hypotheses[0].text stays consistent with the 1-best text above.
      for (auto &hyp : r.hypotheses) {
        hyp.text = ApplyInverseTextNormalization(std::move(hyp.text));
        hyp.text = ApplyHomophoneReplacer(std::move(hyp.text));
      }

      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

  void InitHotwords() {
    // each line in hotwords_file contains space-separated words

    auto is = OpenInputFile(config_.hotwords_file);
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

  template <typename Manager>
  void InitHotwords(Manager *mgr) {
    // each line in hotwords_file contains space-separated words

    auto buf = ReadFile(mgr, config_.hotwords_file);

    std::istringstream is(std::string(buf.begin(), buf.end()));

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
  std::unique_ptr<OfflineTransducerModel> model_;
  std::unique_ptr<OfflineTransducerDecoder> decoder_;
  std::unique_ptr<OfflineLM> lm_;
  int32_t unk_id_ = -1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
