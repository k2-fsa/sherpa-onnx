// sherpa-onnx/csrc/online-recognizer-transducer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_IMPL_H_

#include <algorithm>
#include <memory>
#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-lm.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"
#include "sherpa-onnx/csrc/online-transducer-modified-beam-search-decoder.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

static OnlineRecognizerResult Convert(const OnlineTransducerDecoderResult &src,
                                      const SymbolTable &sym_table,
                                      float frame_shift_ms,
                                      int32_t subsampling_factor,
                                      int32_t segment,
                                      int32_t frames_since_start) {
  OnlineRecognizerResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.tokens.size());

  for (auto i : src.tokens) {
    auto sym = sym_table[i];

    r.text.append(sym);
    r.tokens.push_back(std::move(sym));
  }

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  r.segment = segment;
  r.start_time = frames_since_start * frame_shift_ms / 1000.;

  return r;
}

class OnlineRecognizerTransducerImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerTransducerImpl(const OnlineRecognizerConfig &config)
      : config_(config),
        model_(OnlineTransducerModel::Create(config.model_config)),
        sym_(config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (!config_.hotwords_file.empty()) {
      InitHotwords();
    }
    if (sym_.contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    if (config.decoding_method == "modified_beam_search") {
      if (!config_.lm_config.model.empty()) {
        lm_ = OnlineLM::Create(config.lm_config);
      }

      decoder_ = std::make_unique<OnlineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale, unk_id_);
    } else if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchDecoder>(
          model_.get(), unk_id_);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }
  }

#if __ANDROID_API__ >= 9
  explicit OnlineRecognizerTransducerImpl(AAssetManager *mgr,
                                          const OnlineRecognizerConfig &config)
      : config_(config),
        model_(OnlineTransducerModel::Create(mgr, config.model_config)),
        sym_(mgr, config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (sym_.contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    if (config.decoding_method == "modified_beam_search") {
      decoder_ = std::make_unique<OnlineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale, unk_id_);
    } else if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchDecoder>(
          model_.get(), unk_id_);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }
  }
#endif

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream =
        std::make_unique<OnlineStream>(config_.feat_config, hotwords_graph_);
    InitOnlineStream(stream.get());
    return stream;
  }

  std::unique_ptr<OnlineStream> CreateStream(
      const std::string &hotwords) const override {
    auto hws = std::regex_replace(hotwords, std::regex("/"), "\n");
    std::istringstream is(hws);
    std::vector<std::vector<int32_t>> current;
    if (!EncodeHotwords(is, sym_, &current)) {
      SHERPA_ONNX_LOGE("Encode hotwords failed, skipping, hotwords are : %s",
                       hotwords.c_str());
    }
    current.insert(current.end(), hotwords_.begin(), hotwords_.end());
    auto context_graph =
        std::make_shared<ContextGraph>(current, config_.hotwords_score);
    auto stream =
        std::make_unique<OnlineStream>(config_.feat_config, context_graph);
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

    std::vector<OnlineTransducerDecoderResult> results(n);
    std::vector<float> features_vec(n * chunk_size * feature_dim);
    std::vector<std::vector<Ort::Value>> states_vec(n);
    std::vector<int64_t> all_processed_frames(n);
    bool has_context_graph = false;

    for (int32_t i = 0; i != n; ++i) {
      if (!has_context_graph && ss[i]->GetContextGraph())
        has_context_graph = true;

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

    if (has_context_graph) {
      decoder_->Decode(std::move(pair.first), ss, &results);
    } else {
      decoder_->Decode(std::move(pair.first), &results);
    }

    std::vector<std::vector<Ort::Value>> next_states =
        model_->UnStackStates(pair.second);

    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(results[i]);
      ss[i]->SetStates(std::move(next_states[i]));
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    OnlineTransducerDecoderResult decoder_result = s->GetResult();
    decoder_->StripLeadingBlanks(&decoder_result);

    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    return Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor,
                   s->GetCurrentSegment(), s->GetNumFramesSinceStart());
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
    {
      // segment is incremented only when the last
      // result is not empty
      const auto &r = s->GetResult();
      if (!r.tokens.empty() && r.tokens.back() != 0) {
        s->GetCurrentSegment() += 1;
      }
    }

    // we keep the decoder_out
    decoder_->UpdateDecoderOut(&s->GetResult());
    Ort::Value decoder_out = std::move(s->GetResult().decoder_out);

    auto r = decoder_->GetEmptyResult();
    if (config_.decoding_method == "modified_beam_search" &&
        nullptr != s->GetContextGraph()) {
      for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
        it->second.context_state = s->GetContextGraph()->Root();
      }
    }
    s->SetResult(r);
    s->GetResult().decoder_out = std::move(decoder_out);

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

  void InitHotwords() {
    // each line in hotwords_file contains space-separated words

    std::ifstream is(config_.hotwords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: %s",
                       config_.hotwords_file.c_str());
      exit(-1);
    }

    if (!EncodeHotwords(is, sym_, &hotwords_)) {
      SHERPA_ONNX_LOGE("Encode hotwords failed.");
      exit(-1);
    }
    hotwords_graph_ =
        std::make_shared<ContextGraph>(hotwords_, config_.hotwords_score);
  }

 private:
  void InitOnlineStream(OnlineStream *stream) const {
    auto r = decoder_->GetEmptyResult();

    if (config_.decoding_method == "modified_beam_search" &&
        nullptr != stream->GetContextGraph()) {
      // r.hyps has only one element.
      for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
        it->second.context_state = stream->GetContextGraph()->Root();
      }
    }

    stream->SetResult(r);
    stream->SetStates(model_->GetEncoderInitStates());
  }

 private:
  OnlineRecognizerConfig config_;
  std::vector<std::vector<int32_t>> hotwords_;
  ContextGraphPtr hotwords_graph_;
  std::unique_ptr<OnlineTransducerModel> model_;
  std::unique_ptr<OnlineLM> lm_;
  std::unique_ptr<OnlineTransducerDecoder> decoder_;
  SymbolTable sym_;
  Endpoint endpoint_;
  int32_t unk_id_ = -1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
