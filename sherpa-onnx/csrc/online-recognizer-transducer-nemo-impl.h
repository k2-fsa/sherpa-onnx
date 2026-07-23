// sherpa-onnx/csrc/online-recognizer-transducer-nemo-impl.h
//
// Copyright (c)  2022-2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_

#include <algorithm>
#include <cctype>
#include <fstream>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

// defined in ./online-recognizer-transducer-impl.h
OnlineRecognizerResult Convert(const OnlineTransducerDecoderResult &src,
                               const SymbolTable &sym_table,
                               float frame_shift_ms, int32_t subsampling_factor,
                               int32_t segment, int32_t frames_since_start);

class OnlineRecognizerTransducerNeMoImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerTransducerNeMoImpl(
      const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        endpoint_(config_.endpoint_config),
        model_(
            std::make_unique<OnlineTransducerNeMoModel>(config.model_config)) {
    if (!config.model_config.tokens_buf.empty()) {
      symbol_table_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      /// assuming tokens_buf and tokens are guaranteed not being both empty
      symbol_table_ = SymbolTable(config.model_config.tokens, true);
    }

    if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    PostInit();
  }

  template <typename Manager>
  explicit OnlineRecognizerTransducerNeMoImpl(
      Manager *mgr, const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        endpoint_(config_.endpoint_config),
        model_(std::make_unique<OnlineTransducerNeMoModel>(
            mgr, config.model_config)) {
    if (!config.model_config.tokens_buf.empty()) {
      symbol_table_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      symbol_table_ = SymbolTable(mgr, config.model_config.tokens);
    }
    if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    PostInit();
  }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStream>(config_.feat_config);
    InitOnlineStream(stream.get());
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = model_->SubsamplingFactor();
    const auto &decoder_result = s->GetResult();
    bool has_language_tag =
        !language_tag_token_ids_.empty() && ContainsLanguageTag(decoder_result);
    auto r = has_language_tag
                 ? Convert(FilterLanguageTags(decoder_result), symbol_table_,
                           frame_shift_ms, subsampling_factor,
                           s->GetCurrentSegment(), s->GetNumFramesSinceStart())
                 : Convert(decoder_result, symbol_table_, frame_shift_ms,
                           subsampling_factor, s->GetCurrentSegment(),
                           s->GetNumFramesSinceStart());

    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    return r;
  }

  bool IsEndpoint(OnlineStream *s) const override {
    if (!config_.enable_endpoint) {
      return false;
    }

    int32_t num_processed_frames = s->GetNumProcessedFrames();

    // frame shift is 10 milliseconds
    float frame_shift_in_seconds = 0.01;

    int32_t trailing_silence_frames =
        s->GetResult().num_trailing_blanks * model_->SubsamplingFactor();

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    {
      // segment is incremented only when the last
      // result is not empty
      const auto &r = s->GetResult();
      if (!r.tokens.empty()) {
        s->GetCurrentSegment() += 1;
      }
    }

    s->SetResult({});

    s->SetStates(model_->GetEncoderInitStates());

    s->SetNeMoDecoderStates(model_->GetDecoderInitStates());
    s->SetNeMoDecoderOut(nullptr);

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feature_dim = ss[0]->FeatureDim();

    std::vector<float> features_vec(n * chunk_size * feature_dim);
    std::vector<std::vector<Ort::Value>> encoder_states(n);

    for (int32_t i = 0; i != n; ++i) {
      const auto num_processed_frames = ss[i]->GetNumProcessedFrames();
      std::vector<float> features =
          ss[i]->GetFrames(num_processed_frames, chunk_size);

      // Question: should num_processed_frames include chunk_shift?
      ss[i]->GetNumProcessedFrames() += chunk_shift;

      std::copy(features.begin(), features.end(),
                features_vec.data() + i * chunk_size * feature_dim);

      encoder_states[i] = std::move(ss[i]->GetStates());
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{n, chunk_size, feature_dim};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, features_vec.data(),
                                            features_vec.size(), x_shape.data(),
                                            x_shape.size());

    auto states = model_->StackStates(std::move(encoder_states));
    int32_t num_states = states.size();  // num_states = 3
    auto language_prompt_ids = GetLanguagePromptIds(ss, n);
    auto t = model_->RunEncoder(std::move(x), std::move(states),
                                language_prompt_ids);
    // t[0] encoder_out, float tensor, (batch_size, dim, T)
    // t[1] next states

    std::vector<Ort::Value> out_states;
    out_states.reserve(num_states);

    for (int32_t k = 1; k != num_states + 1; ++k) {
      out_states.push_back(std::move(t[k]));
    }

    auto unstacked_states = model_->UnStackStates(std::move(out_states));
    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetStates(std::move(unstacked_states[i]));
    }

    Ort::Value encoder_out = Transpose12(model_->Allocator(), &t[0]);

    decoder_->Decode(std::move(encoder_out), ss, n);
  }

  void InitOnlineStream(OnlineStream *stream) const {
    // set encoder states
    stream->SetStates(model_->GetEncoderInitStates());

    // set decoder states
    stream->SetNeMoDecoderStates(model_->GetDecoderInitStates());
  }

 private:
  static bool IsLanguageTagToken(const std::string &sym) {
    if (sym.size() < 4 || sym.front() != '<' || sym.back() != '>') {
      return false;
    }

    size_t i = 1;
    int32_t num_lowercase = 0;
    while (i + 1 < sym.size() && num_lowercase != 3 &&
           std::islower(static_cast<unsigned char>(sym[i]))) {
      ++i;
      ++num_lowercase;
    }

    if (num_lowercase < 2) {
      return false;
    }

    if (i == sym.size() - 1) {
      return true;
    }

    if (sym[i] != '-' || i + 3 != sym.size() - 1) {
      return false;
    }

    return std::isupper(static_cast<unsigned char>(sym[i + 1])) &&
           std::isupper(static_cast<unsigned char>(sym[i + 2]));
  }

  void InitLanguageTagTokenIds() {
    if (!model_->IsMultilingual()) {
      return;
    }

    for (int32_t i = 0; i != symbol_table_.NumSymbols(); ++i) {
      if (symbol_table_.Contains(i) && IsLanguageTagToken(symbol_table_[i])) {
        language_tag_token_ids_.insert(i);
      }
    }
  }

  bool ContainsLanguageTag(const OnlineTransducerDecoderResult &src) const {
    for (auto token : src.tokens) {
      if (language_tag_token_ids_.count(token)) {
        return true;
      }
    }

    return false;
  }

  OnlineTransducerDecoderResult FilterLanguageTags(
      const OnlineTransducerDecoderResult &src) const {
    OnlineTransducerDecoderResult ans;
    ans.frame_offset = src.frame_offset;
    ans.num_trailing_blanks = src.num_trailing_blanks;
    ans.tokens.reserve(src.tokens.size());
    ans.timestamps.reserve(src.timestamps.size());

    bool filter_ys_probs = src.ys_probs.size() == src.tokens.size();
    bool filter_lm_probs = src.lm_probs.size() == src.tokens.size();
    bool filter_context_scores = src.context_scores.size() == src.tokens.size();

    if (filter_ys_probs) {
      ans.ys_probs.reserve(src.ys_probs.size());
    } else {
      ans.ys_probs = src.ys_probs;
    }

    if (filter_lm_probs) {
      ans.lm_probs.reserve(src.lm_probs.size());
    } else {
      ans.lm_probs = src.lm_probs;
    }

    if (filter_context_scores) {
      ans.context_scores.reserve(src.context_scores.size());
    } else {
      ans.context_scores = src.context_scores;
    }

    for (size_t i = 0; i != src.tokens.size(); ++i) {
      if (language_tag_token_ids_.count(src.tokens[i])) {
        continue;
      }

      ans.tokens.push_back(src.tokens[i]);
      if (i < src.timestamps.size()) {
        ans.timestamps.push_back(src.timestamps[i]);
      }

      if (filter_ys_probs) {
        ans.ys_probs.push_back(src.ys_probs[i]);
      }
      if (filter_lm_probs) {
        ans.lm_probs.push_back(src.lm_probs[i]);
      }
      if (filter_context_scores) {
        ans.context_scores.push_back(src.context_scores[i]);
      }
    }

    return ans;
  }

  std::vector<int64_t> GetLanguagePromptIds(OnlineStream **ss,
                                            int32_t n) const {
    std::vector<int64_t> ans;
    if (!model_->IsMultilingual()) {
      return ans;
    }

    ans.reserve(n);
    for (int32_t i = 0; i != n; ++i) {
      ans.push_back(model_->GetLanguagePromptId(ss[i]->GetOption("language")));
    }

    return ans;
  }

  void PostInit() {
    config_.feat_config.feature_dim = model_->FeatureDim();

    config_.feat_config.low_freq = 0;
    config_.feat_config.high_freq = 8000;
    config_.feat_config.is_librosa = true;
    config_.feat_config.remove_dc_offset = false;
    config_.feat_config.window_type = "hann";
    config_.feat_config.dither = 0;
    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    int32_t vocab_size = model_->VocabSize();

    // check the blank ID
    if (!symbol_table_.Contains("<blk>")) {
      SHERPA_ONNX_LOGE("tokens.txt does not include the blank token <blk>");
      SHERPA_ONNX_EXIT(-1);
    }

    if (symbol_table_["<blk>"] != vocab_size - 1) {
      SHERPA_ONNX_LOGE("<blk> is not the last token!");
      SHERPA_ONNX_EXIT(-1);
    }

    if (symbol_table_.NumSymbols() != vocab_size) {
      SHERPA_ONNX_LOGE("number of lines in tokens.txt %d != %d (vocab_size)",
                       symbol_table_.NumSymbols(), vocab_size);
      SHERPA_ONNX_EXIT(-1);
    }

    InitLanguageTagTokenIds();
  }

 private:
  OnlineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OnlineTransducerNeMoModel> model_;
  std::unordered_set<int64_t> language_tag_token_ids_;
  std::unique_ptr<OnlineTransducerGreedySearchNeMoDecoder> decoder_;
  Endpoint endpoint_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_
