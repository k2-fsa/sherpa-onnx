// sherpa-onnx/csrc/rknn/keyword-spotter-transducer-rknn-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_KEYWORD_SPOTTER_TRANSDUCER_RKNN_IMPL_H_
#define SHERPA_ONNX_CSRC_RKNN_KEYWORD_SPOTTER_TRANSDUCER_RKNN_IMPL_H_

#include <algorithm>
#include <memory>
#include <regex>  // NOLINT
#include <string>
#include <strstream>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/keyword-spotter-impl.h"
#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/rknn/online-stream-rknn.h"
#include "sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.h"
#include "sherpa-onnx/csrc/rknn/transducer-keyword-decoder-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

KeywordResult Convert(const TransducerKeywordResult &src,
                      const SymbolTable &sym_table, float frame_shift_ms,
                      int32_t subsampling_factor, int32_t frames_since_start);

class KeywordSpotterTransducerRknnImpl : public KeywordSpotterImpl {
 public:
  explicit KeywordSpotterTransducerRknnImpl(const KeywordSpotterConfig &config)
      : config_(config),
        model_(std::make_unique<OnlineZipformerTransducerModelRknn>(
            config.model_config)) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      /// assuming tokens_buf and tokens are guaranteed not being both empty
      sym_ = SymbolTable(config.model_config.tokens, true);
    }

    if (sym_.Contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    if (config.keywords_buf.empty()) {
      InitKeywords();
    } else {
      InitKeywordsFromBufStr();
    }

    decoder_ = std::make_unique<TransducerKeywordDecoderRknn>(
        model_.get(), config_.max_active_paths, config_.num_trailing_blanks,
        unk_id_);
  }

  template <typename Manager>
  KeywordSpotterTransducerRknnImpl(Manager *mgr,
                                   const KeywordSpotterConfig &config)
      : config_(config),
        model_(std::make_unique<OnlineZipformerTransducerModelRknn>(
            mgr, config.model_config)),
        sym_(mgr, config.model_config.tokens) {
    if (sym_.Contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    InitKeywords(mgr);

    decoder_ = std::make_unique<TransducerKeywordDecoderRknn>(
        model_.get(), config_.max_active_paths, config_.num_trailing_blanks,
        unk_id_);
  }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStreamRknn>(config_.feat_config,
                                                     keywords_graph_);

    InitOnlineStream(stream.get());
    return stream;
  }

  std::unique_ptr<OnlineStream> CreateStream(
      const std::string &keywords) const override {
    auto kws = std::regex_replace(keywords, std::regex("/"), "\n");
    std::istringstream is(kws);

    std::vector<std::vector<int32_t>> current_ids;
    std::vector<std::string> current_kws;
    std::vector<float> current_scores;
    std::vector<float> current_thresholds;

    if (!EncodeKeywords(is, sym_, &current_ids, &current_kws, &current_scores,
                        &current_thresholds)) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Encode keywords %{public}s failed.", keywords.c_str());
#else
      SHERPA_ONNX_LOGE("Encode keywords %s failed.", keywords.c_str());
#endif
      return nullptr;
    }

    int32_t num_kws = current_ids.size();
    int32_t num_default_kws = keywords_id_.size();

    current_ids.insert(current_ids.end(), keywords_id_.begin(),
                       keywords_id_.end());

    if (!current_kws.empty() && !keywords_.empty()) {
      current_kws.insert(current_kws.end(), keywords_.begin(), keywords_.end());
    } else if (!current_kws.empty() && keywords_.empty()) {
      current_kws.insert(current_kws.end(), num_default_kws, std::string());
    } else if (current_kws.empty() && !keywords_.empty()) {
      current_kws.insert(current_kws.end(), num_kws, std::string());
      current_kws.insert(current_kws.end(), keywords_.begin(), keywords_.end());
    } else {
      // Do nothing.
    }

    if (!current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else if (!current_scores.empty() && boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_default_kws,
                            config_.keywords_score);
    } else if (current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_kws,
                            config_.keywords_score);
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else {
      // Do nothing.
    }

    if (!current_thresholds.empty() && !thresholds_.empty()) {
      current_thresholds.insert(current_thresholds.end(), thresholds_.begin(),
                                thresholds_.end());
    } else if (!current_thresholds.empty() && thresholds_.empty()) {
      current_thresholds.insert(current_thresholds.end(), num_default_kws,
                                config_.keywords_threshold);
    } else if (current_thresholds.empty() && !thresholds_.empty()) {
      current_thresholds.insert(current_thresholds.end(), num_kws,
                                config_.keywords_threshold);
      current_thresholds.insert(current_thresholds.end(), thresholds_.begin(),
                                thresholds_.end());
    } else {
      // Do nothing.
    }

    auto keywords_graph = std::make_shared<ContextGraph>(
        current_ids, config_.keywords_score, config_.keywords_threshold,
        current_scores, current_kws, current_thresholds);

    auto stream =
        std::make_unique<OnlineStreamRknn>(config_.feat_config, keywords_graph);
    InitOnlineStream(stream.get());
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  void Reset(OnlineStream *s) const override {
    InitOnlineStream(reinterpret_cast<OnlineStreamRknn *>(s));
  }

  void DecodeStream(OnlineStreamRknn *s) const {
    auto r = s->GetKeywordResult(true);
    int32_t num_trailing_blanks = r.num_trailing_blanks;
    // assume subsampling_factor is 4
    // assume frameshift is 0.01 second
    float trailing_slience = num_trailing_blanks * 4 * 0.01;

    // it resets automatically after detecting 1.5 seconds of silence
    float threshold = 1.5;
    if (trailing_slience > threshold) {
      Reset(s);
    }

    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feature_dim = s->FeatureDim();

    const auto num_processed_frames = s->GetNumProcessedFrames();

    std::vector<float> features =
        s->GetFrames(num_processed_frames, chunk_size);
    s->GetNumProcessedFrames() += chunk_shift;

    auto &states = s->GetZipformerEncoderStates();

    auto p = model_->RunEncoder(features, std::move(states));

    states = std::move(p.second);

    decoder_->Decode(std::move(p.first), s);
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeStream(reinterpret_cast<OnlineStreamRknn *>(ss[i]));
    }
  }

  KeywordResult GetResult(OnlineStream *s) const override {
    TransducerKeywordResult decoder_result = s->GetKeywordResult(true);

    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    return Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor,
                   s->GetNumFramesSinceStart());
  }

 private:
  void InitKeywords(std::istream &is) {
    if (!EncodeKeywords(is, sym_, &keywords_id_, &keywords_, &boost_scores_,
                        &thresholds_)) {
      SHERPA_ONNX_LOGE("Encode keywords failed.");
      exit(-1);
    }
    keywords_graph_ = std::make_shared<ContextGraph>(
        keywords_id_, config_.keywords_score, config_.keywords_threshold,
        boost_scores_, keywords_, thresholds_);
  }

  void InitKeywords() {
#ifdef SHERPA_ONNX_ENABLE_WASM_KWS
    // Due to the limitations of the wasm file system,
    // the keyword_file variable is directly parsed as a string of keywords
    // if WASM KWS on
    std::istringstream is(config_.keywords_file);
    InitKeywords(is);
#else
    // each line in keywords_file contains space-separated words
    std::ifstream is(config_.keywords_file);
    if (!is) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Open keywords file failed: %{public}s",
                       config_.keywords_file.c_str());
#else
      SHERPA_ONNX_LOGE("Open keywords file failed: %s",
                       config_.keywords_file.c_str());
#endif
      exit(-1);
    }
    InitKeywords(is);
#endif
  }

  template <typename Manager>
  void InitKeywords(Manager *mgr) {
    // each line in keywords_file contains space-separated words

    auto buf = ReadFile(mgr, config_.keywords_file);

    std::istrstream is(buf.data(), buf.size());

    if (!is) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Open keywords file failed: %{public}s",
                       config_.keywords_file.c_str());
#else
      SHERPA_ONNX_LOGE("Open keywords file failed: %s",
                       config_.keywords_file.c_str());
#endif
      exit(-1);
    }
    InitKeywords(is);
  }

  void InitKeywordsFromBufStr() {
    // keywords_buf's content is supposed to be same as the keywords_file's
    std::istringstream is(config_.keywords_buf);
    InitKeywords(is);
  }

  void InitOnlineStream(OnlineStreamRknn *stream) const {
    auto r = decoder_->GetEmptyResult();
    SHERPA_ONNX_CHECK_EQ(r.hyps.Size(), 1);

    SHERPA_ONNX_CHECK(stream->GetContextGraph() != nullptr);
    r.hyps.begin()->second.context_state = stream->GetContextGraph()->Root();

    stream->SetKeywordResult(r);
    stream->SetZipformerEncoderStates(model_->GetEncoderInitStates());
  }

 private:
  KeywordSpotterConfig config_;
  std::vector<std::vector<int32_t>> keywords_id_;
  std::vector<float> boost_scores_;
  std::vector<float> thresholds_;
  std::vector<std::string> keywords_;
  ContextGraphPtr keywords_graph_;
  std::unique_ptr<OnlineZipformerTransducerModelRknn> model_;

  std::unique_ptr<TransducerKeywordDecoderRknn> decoder_;
  SymbolTable sym_;
  int32_t unk_id_ = -1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_KEYWORD_SPOTTER_TRANSDUCER_RKNN_IMPL_H_
