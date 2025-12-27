// sherpa-onnx/csrc/offline-recognizer-whisper-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-whisper-decoder.h"
#include "sherpa-onnx/csrc/offline-whisper-dtw.h"
#include "sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-whisper-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineRecognizerWhisperImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerWhisperImpl(const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineWhisperModel>(config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerWhisperImpl(Manager *mgr,
                               const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(
            std::make_unique<OfflineWhisperModel>(mgr, config.model_config)) {
    Init();
  }

  void Init() {
    // tokens.txt from whisper is base64 encoded, so we need to decode it
    symbol_table_.ApplyBase64Decode();

    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineWhisperGreedySearchDecoder>(
          config_.model_config.whisper, model_.get());
    } else {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported at present for whisper. Given %s",
          config_.decoding_method.c_str());
      exit(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    WhisperTag tag;
    tag.dim = model_->FeatureDim();
    return std::make_unique<OfflineStream>(tag);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    // batch decoding is not implemented yet
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  void SetConfig(const OfflineRecognizerConfig &config) override {
    config_.model_config.whisper = config.model_config.whisper;
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeStream(OfflineStream *s) const {
    decoder_->SetConfig(config_.model_config.whisper);

    int32_t max_num_frames = 3000;
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();
    int32_t num_frames = f.size() / feat_dim;

    // we use 50 here so that there will be some zero tail paddings
    if (num_frames >= max_num_frames - 50) {
      SHERPA_ONNX_LOGE(
          "Only waves less than 30 seconds are supported. We process only the "
          "first 30 seconds and discard the remaining data");
      num_frames = max_num_frames - 50;
    }

    model_->NormalizeFeatures(f.data(), num_frames, feat_dim);

    // note that 1000 is an experience-value.
    // You can replace 1000 by other values, say, 100.
    //
    // Since we have removed the 30 seconds constraint, we need
    // tail_padding_frames so that whisper is able to detect the eot token.
    int32_t tail_padding_frames = 1000;

    if (config_.model_config.whisper.tail_paddings > 0) {
      tail_padding_frames = config_.model_config.whisper.tail_paddings;
    }

    int32_t actual_frames =
        std::min(num_frames + tail_padding_frames, max_num_frames);

    std::array<int64_t, 3> shape{1, actual_frames, feat_dim};

    Ort::Value mel = Ort::Value::CreateTensor<float>(
        model_->Allocator(), shape.data(), shape.size());

    float *p_mel = mel.GetTensorMutableData<float>();
    std::copy(f.data(), f.data() + num_frames * feat_dim, p_mel);

    std::fill_n(p_mel + num_frames * feat_dim,
                (actual_frames - num_frames) * feat_dim, 0);

    mel = Transpose12(model_->Allocator(), &mel);

    try {
      auto cross_kv = model_->ForwardEncoder(std::move(mel));

      auto results = decoder_->Decode(std::move(cross_kv.first),
                                      std::move(cross_kv.second), num_frames);

      auto r = Convert(results[0], symbol_table_);
      s->SetResult(r);
    } catch (const Ort::Exception &ex) {
      SHERPA_ONNX_LOGE(
          "\n\nCaught exception:\n\n%s\n\nReturn an empty result. Number of "
          "input frames: %d, Current tail "
          "paddings: %d. If you see a lot of such exceptions, please consider "
          "using a larger --whisper-tail-paddings",
          ex.what(), num_frames, tail_padding_frames);
      return;
    }
  }

 private:
  OfflineRecognitionResult Convert(const OfflineWhisperDecoderResult &src,
                                   const SymbolTable &sym_table) const {
    OfflineRecognitionResult r;

    bool enable_timestamps = config_.model_config.whisper.enable_timestamps;

    std::string text;

    // Since we always use no_timestamps mode, src.tokens contains only text tokens
    for (size_t i = 0; i < src.tokens.size(); ++i) {
      int32_t token_id = src.tokens[i];

      if (!sym_table.Contains(token_id)) {
        continue;
      }

      std::string s = sym_table[token_id];
      s = ApplyInverseTextNormalization(s);
      s = ApplyHomophoneReplacer(std::move(s));

      text += s;
      r.tokens.push_back(s);
    }

    r.text = text;
    r.lang = src.lang;

    // Compute token-level and word-level timestamps using DTW if enabled
    if (enable_timestamps && !src.attention_weights.empty() &&
        !r.tokens.empty()) {
      ComputeTimestamps(src, r);
    }

    return r;
  }

  // Compute token-level and word-level timestamps using cross-attention DTW
  void ComputeTimestamps(const OfflineWhisperDecoderResult &src,
                         OfflineRecognitionResult &r) const {
    // Compute word boundaries from tokens
    auto word_boundaries = ComputeWordBoundaries(r.tokens);

    // Compute DTW alignment
    WhisperDTW dtw;

    // Note: src.attention includes all tokens (initial + decoded)
    // The first few are SOT sequence tokens which DTW will skip.
    // Initial tokens are: [sot, lang, task, no_timestamps] for multilingual,
    // or [sot, no_timestamps] for English-only models.
    // We skip sot_sequence (without no_timestamps), keeping no_timestamps as
    // the first token in DTW - it acts as a start anchor like in OpenAI's code.
    int32_t sot_sequence_length =
        static_cast<int32_t>(model_->GetInitialTokens().size());

    std::vector<int32_t> token_frames = dtw.ComputeAlignment(
        src.attention_weights.data(), src.attention_n_heads,
        src.attention_n_tokens, src.attention_n_frames, src.num_audio_frames,
        sot_sequence_length);

    // Convert frame indices to timestamps
    std::vector<float> token_times =
        WhisperDTW::FrameIndicesToSeconds(token_frames);

    // Populate token-level timestamps
    // token_times[0] is the anchor (no_timestamps), token_times[i+1] is for text token i
    r.timestamps.clear();
    r.timestamps.reserve(r.tokens.size());
    for (size_t i = 0; i < r.tokens.size(); ++i) {
      // Use anchor time (index 0) for first token, otherwise use token's own time
      int32_t time_idx = static_cast<int32_t>(i);
      if (time_idx < static_cast<int32_t>(token_times.size())) {
        r.timestamps.push_back(token_times[time_idx]);
      } else {
        r.timestamps.push_back(r.timestamps.empty() ? 0.0f : r.timestamps.back());
      }
    }

    // Populate word-level timestamps
    for (const auto &wb : word_boundaries) {
      OfflineRecognitionWordTiming wt;
      wt.word = wb.word;
      wt.probability = 0.0f;  // We don't have confidence from DTW

      // Word start time
      if (wb.start_token >= 0 &&
          wb.start_token < static_cast<int32_t>(token_times.size())) {
        wt.start = token_times[wb.start_token];
      } else {
        wt.start = 0.0f;
      }

      // Word end time (start of next token after last token in word)
      if (wb.end_token > 0 &&
          wb.end_token < static_cast<int32_t>(token_times.size())) {
        wt.end = token_times[wb.end_token];
      } else if (wb.end_token > 0 && wb.end_token - 1 >= 0 &&
                 wb.end_token - 1 < static_cast<int32_t>(token_times.size())) {
        // Use last token's time + one frame duration
        wt.end = token_times[wb.end_token - 1] + kWhisperSecondsPerToken;
      } else {
        wt.end = wt.start + kWhisperSecondsPerToken;
      }

      r.word_timestamps.push_back(std::move(wt));
    }
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineWhisperModel> model_;
  std::unique_ptr<OfflineWhisperDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
