// sherpa-onnx/csrc/keyword-spotter-transducer-impl.h
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_TRANSDUCER_IMPL_H_
#define SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_TRANSDUCER_IMPL_H_

#include <algorithm>
#include <memory>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/keyword-spotter-impl.h"
#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transducer-keywords-decoder.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

static KeywordResult Convert(const TransducerKeywordsDecoderResult &src,
                             const SymbolTable &sym_table, float frame_shift_ms,
                             int32_t subsampling_factor,
                             int32_t frames_since_start) {
  KeywordResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.tokens.size());
  r.keyword = src.keyword;

  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    r.tokens.push_back(std::move(sym));
  }

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  r.start_time = frames_since_start * frame_shift_ms / 1000.;

  return r;
}

class KeywordSpotterTransducerImpl : public KeywordSpotterImpl {
 public:
  explicit KeywordSpotterTransducerImpl(const KeywordSpotterConfig &config)
      : config_(config),
        model_(OnlineTransducerModel::Create(config.model_config)),
        sym_(config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (sym_.contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    InitKeywords();

    decoder_ = std::make_unique<TransducerKeywordsDecoder>(
        model_.get(), config_.max_active_paths, config_.num_tailing_blanks);
  }

#if __ANDROID_API__ >= 9
  explicit KeywordSpotterTransducerImpl(AAssetManager *mgr,
                                        const KeywordSpotterConfig &config)
      : config_(config),
        model_(OnlineTransducerModel::Create(mgr, config.model_config)),
        sym_(mgr, config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (sym_.contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    InitKeywords(mgr);

    decoder_ = std::make_unique<TransducerKeywordsDecoder>(
        model_.get(), config_.max_active_paths, config_.num_tailing_blanks);
  }
#endif

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream =
        std::make_unique<OnlineStream>(config_.feat_config, keywords_graph_);
    InitOnlineStream(stream.get());
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feature_dim = ss[0]->FeatureDim();

    std::vector<TransducerKeywordsDecoderResult> results(n);
    std::vector<float> features_vec(n * chunk_size * feature_dim);
    std::vector<std::vector<Ort::Value>> states_vec(n);
    std::vector<int64_t> all_processed_frames(n);

    for (int32_t i = 0; i != n; ++i) {
      SHERPA_ONNX_CHECK(ss[i]->GetContextGraph() != nullptr);

      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> features =
          ss[i]->GetFrames(num_processed_frames, chunk_size);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;

      std::copy(features.begin(), features.end(),
                features_vec.data() + i * chunk_size * feature_dim);

      results[i] = std::move(ss[i]->GetResult());
      states_vec[i] = std::move(ss[i]->GetStates());
      all_processed_frames[i] = num_processed_frames;
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{n, chunk_size, feature_dim};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, features_vec.data(),
                                            features_vec.size(), x_shape.data(),
                                            x_shape.size());

    std::array<int64_t, 1> processed_frames_shape{
        static_cast<int64_t>(all_processed_frames.size())};

    Ort::Value processed_frames = Ort::Value::CreateTensor(
        memory_info, all_processed_frames.data(), all_processed_frames.size(),
        processed_frames_shape.data(), processed_frames_shape.size());

    auto states = model_->StackStates(states_vec);

    auto pair = model_->RunEncoder(std::move(x), std::move(states),
                                   std::move(processed_frames));

    decoder_->Decode(std::move(pair.first), ss, &results);

    std::vector<std::vector<Ort::Value>> next_states =
        model_->UnStackStates(pair.second);

    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(results[i]);
      ss[i]->SetStates(std::move(next_states[i]));
    }
  }

  KeywordResult GetResult(OnlineStream *s) const override {
    TransducerKeywordsResult decoder_result = s->GetResult();

    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    return Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor,
                   s->GetNumFramesSinceStart());
  }

  bool IsEndpoint(OnlineStream *s) const override {
    if (!config_.enable_endpoint) {
      return false;
    }

    int32_t num_processed_frames = s->GetNumProcessedFrames();

    // frame shift is 10 milliseconds
    float frame_shift_in_seconds = 0.01;

    // subsampling factor is 4
    int32_t trailing_silence_frames = s->GetResult().num_trailing_blanks * 4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    auto r = decoder_->GetEmptyResult();
    SHERPA_ONNX_CHECK_EQ(r.hyps.size(), 1);
    r.hyps.begin()->second.context_state = stream->GetContextGraph()->Root();

    s->SetResult(r);

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

 private:
  void InitKeywords() {
    // each line in keywords_file contains space-separated words

    std::ifstream is(config_.keywords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open keywords file failed: %s",
                       config_.keywords_file.c_str());
      exit(-1);
    }

    if (!EncodeKeywords(is, sym_, &keywords_id_, &keywords_, &boost_scores_,
                        &thresholds_)) {
      SHERPA_ONNX_LOGE("Encode keywords failed.");
      exit(-1);
    }
    keywords_graph_ = std::make_shared<ContextGraph>(
        keywords_id_, config_.keywords_score, config_.keywords_threshold,
        boost_scores_, keywords_, thresholds_);
  }

#if __ANDROID_API__ >= 9
  void InitKeywords(AAssetManager *mgr) {
    // each line in keywords_file contains space-separated words

    auto buf = ReadFile(mgr, config_.keywords_file);

    std::istrstream is(buf.data(), buf.size());

    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: %s",
                       config_.keywords_file.c_str());
      exit(-1);
    }

    if (!EncodeKeywords(is, sym_, &keywords_id_, &keywords_, &boost_scores_,
                        &thresholds_)) {
      SHERPA_ONNX_LOGE("Encode keywords failed.");
      exit(-1);
    }
    keywords_graph_ = std::make_shared<ContextGraph>(
        keywords_id_, config_.keywords_score, config_.keywords_threshold,
        boost_scores_, keywords_, thresholds_);
  }
#endif

  void InitOnlineStream(OnlineStream *stream) const {
    auto r = decoder_->GetEmptyResult();
    SHERPA_ONNX_CHECK_EQ(r.hyps.size(), 1);

    SHERPA_ONNX_CHECK(stream->GetContextGraph() != nullptr);
    r.hyps.begin()->second.context_state = stream->GetContextGraph()->Root();

    stream->SetResult(r);
    stream->SetStates(model_->GetEncoderInitStates());
  }

 private:
  KeywordSpotterConfig config_;
  std::vector<std::vector<int32_t>> keywords_id_;
  std::vector<float> boost_scores_;
  std::vector<float> thresholds_;
  std::vector<std::string> keywords_;
  ContextGraphPtr keywords_graph_;
  std::unique_ptr<OnlineTransducerModel> model_;
  std::unique_ptr<TransducerKeywordsDecoder> decoder_;
  SymbolTable sym_;
  Endpoint endpoint_;
  int32_t unk_id_ = -1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_TRANSDUCER_IMPL_H_
