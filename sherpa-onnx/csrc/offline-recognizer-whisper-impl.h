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
    r.tokens.reserve(src.tokens.size());

    std::string text;

    // Get timestamp begin token ID to filter out timestamp tokens
    int32_t timestamp_begin = model_->TimestampBegin();
    bool enable_segment_timestamps =
        config_.model_config.whisper.enable_segment_timestamps;

    // Build text, skipping timestamp tokens if in segment timestamp mode
    for (auto i : src.tokens) {
      // Skip timestamp tokens (they are >= timestamp_begin)
      if (enable_segment_timestamps && i >= timestamp_begin) {
        continue;
      }

      if (!sym_table.Contains(i)) {
        continue;
      }

      std::string s = sym_table[i];
      s = ApplyInverseTextNormalization(s);
      s = ApplyHomophoneReplacer(std::move(s));

      text += s;
      r.tokens.push_back(s);
    }

    r.text = text;
    r.lang = src.lang;

    // Convert segments from segment timestamp mode to parallel vectors
    if (enable_segment_timestamps && !src.segments.empty()) {
      r.segment_timestamps.reserve(src.segments.size());
      r.segment_durations.reserve(src.segments.size());
      r.segment_texts.reserve(src.segments.size());

      for (const auto &seg : src.segments) {
        r.segment_timestamps.push_back(seg.start_time);
        r.segment_durations.push_back(seg.end_time - seg.start_time);

        // Convert token IDs to text
        std::string seg_text;
        for (int32_t tok_id : seg.token_ids) {
          if (sym_table.Contains(tok_id)) {
            std::string s = sym_table[tok_id];
            s = ApplyInverseTextNormalization(s);
            s = ApplyHomophoneReplacer(std::move(s));
            seg_text += s;
          }
        }
        r.segment_texts.push_back(std::move(seg_text));
      }
    }

    // Compute token-level timestamps using DTW if enabled
    if (config_.model_config.whisper.enable_timestamps &&
        !src.attention_weights.empty() &&
        !r.tokens.empty()) {
      ComputeTimestamps(src, r);
    }

    return r;
  }

  // Compute token-level timestamps using cross-attention DTW
  void ComputeTimestamps(const OfflineWhisperDecoderResult &src,
                         OfflineRecognitionResult &r) const {
    WhisperDTW dtw;

    // Note: src.attention includes all tokens (initial + decoded)
    // The first few are SOT sequence tokens which DTW will skip.
    // Initial tokens are: [sot, lang, task, no_timestamps] for multilingual,
    // or [sot, no_timestamps] for English-only models.
    int32_t sot_sequence_length =
        static_cast<int32_t>(model_->GetInitialTokens().size());

    // Use ComputeTokenTimings which extracts both start times and durations
    // directly from the DTW jump_times, following OpenAI's approach:
    //   start_times[i] = jump_times[i]
    //   end_times[i] = jump_times[i+1]
    //   durations[i] = end_times[i] - start_times[i]
    TokenTimingResult timing = dtw.ComputeTokenTimings(
        src.attention_weights.data(), src.attention_n_heads,
        src.attention_n_tokens, src.attention_n_frames, src.num_audio_frames,
        sot_sequence_length, static_cast<int32_t>(r.tokens.size()));

    // Populate timestamps and durations
    r.timestamps = std::move(timing.start_times);
    r.durations = std::move(timing.durations);

    // Ensure vectors match token count
    if (r.timestamps.size() != r.tokens.size()) {
      SHERPA_ONNX_LOGE(
          "DTW returned %zu timestamps for %zu tokens, padding/truncating",
          r.timestamps.size(), r.tokens.size());
    }
    float fill_time = r.timestamps.empty() ? 0.0f : r.timestamps.back();
    r.timestamps.resize(r.tokens.size(), fill_time);
    r.durations.resize(r.tokens.size(), 0.0f);

    // Clamp token end times to segment boundaries (like OpenAI timing.py)
    // If a token ends more than 0.5s after segment end, truncate it.
    // This prevents DTW-derived timings from extending past segment bounds.
    if (!src.segments.empty() && !r.timestamps.empty()) {
      float segment_end = src.segments.back().end_time;
      if (segment_end > 0) {
        for (size_t i = 0; i < r.timestamps.size(); ++i) {
          float token_end = r.timestamps[i] + r.durations[i];
          // Like OpenAI: if token_end > segment_end + 0.5, clamp it
          if (token_end > segment_end + 0.5f) {
            r.durations[i] = std::max(0.0f, segment_end - r.timestamps[i]);
          }
        }
      }
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
