// sherpa-onnx/csrc/online-recognizer-transducer-nemo-parakeet-unified-impl.h
//
// Copyright (c)  2026  Milan Leonard
// Copyright (c)  2026  Julian Pscheid

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_PARAKEET_UNIFIED_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_PARAKEET_UNIFIED_IMPL_H_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/online-transducer-greedy-search-nemo-parakeet-unified-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-parakeet-unified-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

OnlineRecognizerResult Convert(const OnlineTransducerDecoderResult &src,
                               const SymbolTable &sym_table,
                               float frame_shift_ms, int32_t subsampling_factor,
                               int32_t segment, int32_t frames_since_start,
                               int32_t remove_cjk_space);

class OnlineRecognizerTransducerNeMoParakeetUnifiedImpl
    : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerTransducerNeMoParakeetUnifiedImpl(
      const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        model_(std::make_unique<OnlineTransducerNeMoParakeetUnifiedModel>(
            config.model_config)),
        endpoint_(config_.endpoint_config) {
    if (!config.model_config.tokens_buf.empty()) {
      symbol_table_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      symbol_table_ = SymbolTable(config.model_config.tokens, true);
    }

    BuildDecoder();
    PostInit();
  }

  template <typename Manager>
  explicit OnlineRecognizerTransducerNeMoParakeetUnifiedImpl(
      Manager *mgr, const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        model_(std::make_unique<OnlineTransducerNeMoParakeetUnifiedModel>(
            mgr, config.model_config)),
        endpoint_(config_.endpoint_config) {
    if (!config.model_config.tokens_buf.empty()) {
      symbol_table_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      symbol_table_ = SymbolTable(mgr, config.model_config.tokens);
    }

    BuildDecoder();
    PostInit();
  }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStream>(config_.feat_config);
    InitOnlineStream(stream.get());
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    int32_t processed = s->GetNumProcessedFrames();
    int32_t ready = s->NumFramesReady();

    int32_t required =
        model_->ChunkFeatureFrames() + model_->RightFeatureFrames();
    if (model_->RightFeatureFrames() == 0) {
      // Keep the boundary chunk until another complete hop or EOF. Otherwise
      // pre-EOF decoding can strand a partial hop accepted after that chunk.
      ++required;
    }
    if (ready - processed >= required) {
      return true;
    }

    // Flush the final, possibly short, center and look-ahead after
    // InputFinished().
    return ready > processed && s->IsLastFrame(ready - 1);
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = model_->SubsamplingFactor();
    // Per-stream option "remove_cjk_space": use 1 or 0. 0 means disable,
    // non-zero (default 1) means enable removing spaces between CJK
    // characters of scripts without word spacing.
    auto r = Convert(s->GetResult(), symbol_table_, frame_shift_ms,
                     subsampling_factor, s->GetCurrentSegment(),
                     s->GetNumFramesSinceStart(),
                     s->GetOptionInt("remove_cjk_space", 1));
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    return r;
  }

  bool IsEndpoint(OnlineStream *s) const override {
    if (!config_.enable_endpoint) {
      return false;
    }

    int32_t num_processed_frames = s->GetNumProcessedFrames();
    float frame_shift_in_seconds = 0.01;
    int32_t trailing_silence_frames =
        s->GetResult().num_trailing_blanks * model_->SubsamplingFactor();

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    {
      const auto &r = s->GetResult();
      if (!r.tokens.empty()) {
        s->GetCurrentSegment() += 1;
      }
    }

    s->SetResult({});
    // Endpointing clears segment output without interrupting the buffered
    // encoder and predictor context, as in continuous NeMo decoding.
    s->Reset();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      DecodeOneStream(ss[i]);
    }
  }

  void InitOnlineStream(OnlineStream *stream) const {
    stream->SetNeMoDecoderStates(model_->GetDecoderInitStates());
    stream->SetNeMoDecoderOut(Ort::Value{nullptr});
  }

 private:
  void BuildDecoder() {
    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<
          OnlineTransducerGreedySearchNeMoParakeetUnifiedDecoder>(
          model_.get(), config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void PostInit() {
    config_.feat_config.feature_dim = model_->FeatureDim();
    config_.feat_config.low_freq = 0;
    config_.feat_config.high_freq = 8000;
    config_.feat_config.is_librosa = true;
    config_.feat_config.remove_dc_offset = false;
    config_.feat_config.window_type = "hann";
    config_.feat_config.nemo_buffered_capacity = model_->TotalFeatureFrames();
    config_.feat_config.dither = 0;
    config_.feat_config.nemo_normalize_type = "";

    int32_t vocab_size = model_->VocabSize();
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

    if (model_->FeatureNormalizationMethod() != "per_feature" &&
        !model_->FeatureNormalizationMethod().empty()) {
      SHERPA_ONNX_LOGE("Unsupported NeMo Parakeet Unified normalization: %s",
                       model_->FeatureNormalizationMethod().c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  Ort::Value SliceCenterEncoderOut(Ort::Value *encoder_out, int32_t left_frames,
                                   int32_t center_frames) const {
    auto shape = encoder_out->GetTensorTypeAndShapeInfo().GetShape();
    int32_t num_rows = static_cast<int32_t>(shape[1]);
    int32_t num_cols = static_cast<int32_t>(shape[2]);
    int32_t start = left_frames;
    int32_t end = std::min(start + center_frames, num_rows);

    if (start >= end) {
      SHERPA_ONNX_LOGE(
          "Invalid Parakeet Unified encoder slice: start %d, end %d", start,
          end);
      SHERPA_ONNX_EXIT(-1);
    }

    const float *src = encoder_out->GetTensorData<float>();
    std::array<int64_t, 3> center_shape{1, end - start, num_cols};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    return Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(src) + start * num_cols,
        (end - start) * num_cols, center_shape.data(), center_shape.size());
  }

  void DecodeOneStream(OnlineStream *s) const {
    int32_t processed = s->GetNumProcessedFrames();
    int32_t ready = s->NumFramesReady();
    bool is_last = s->IsLastFrame(ready - 1);
    int32_t remaining = ready - processed;
    int32_t complete_step_frames =
        model_->ChunkFeatureFrames() + model_->RightFeatureFrames();
    bool final_chunk =
        is_last &&
        (remaining < complete_step_frames ||
         (remaining == complete_step_frames &&
          (model_->RightFeatureFrames() == 0 || s->NumTrailingSamples() == 0)));
    // NeMo consumes the final chunk and its look-ahead together.
    int32_t valid_center_frames =
        final_chunk ? remaining
                    : std::min(model_->ChunkFeatureFrames(), remaining);

    if (valid_center_frames <= 0) {
      return;
    }

    int32_t feature_dim = model_->FeatureDim();
    // Keep the encoder's audio context across endpoint resets. The feature
    // extractor survives Reset(), and GetFrames() uses segment-relative
    // indices.
    int32_t right_frames =
        std::min(model_->RightFeatureFrames(), remaining - valid_center_frames);
    // The last NeMo buffer keeps its total capacity: a short final read can
    // leave more than LeftFeatureFrames() of history before the final center.
    int32_t max_left = final_chunk
                           ? model_->TotalFeatureFrames() - valid_center_frames
                           : model_->LeftFeatureFrames();
    int32_t left_frames =
        std::min(max_left, processed + s->GetNumFramesSinceStart());
    int32_t total_feature_frames =
        left_frames + valid_center_frames + right_frames;
    int32_t start_sample_offset = 0;
    std::vector<float> features =
        s->GetBufferedFeatures(processed - left_frames, total_feature_frames,
                               final_chunk, &start_sample_offset);

    if (model_->FeatureNormalizationMethod() == "per_feature") {
      NemoNormalizePerFeature(features.data(), total_feature_frames,
                              feature_dim, /*sample_variance=*/true);
    }

    // NeMo's centered STFT produces one extra feature frame, masked to zero
    // after normalization. It is encoder input padding, not CMVN context.
    features.resize((total_feature_frames + 1) * feature_dim, 0.0f);

    int32_t valid_center_encoder_frames =
        (valid_center_frames + model_->SubsamplingFactor() - 1) /
        model_->SubsamplingFactor();

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{1, total_feature_frames + 1, feature_dim};
    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, features.data(), features.size(),
                                 x_shape.data(), x_shape.size());

    std::array<int64_t, 1> length_shape{1};
    Ort::Value length = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), length_shape.data(), length_shape.size());
    length.GetTensorMutableData<int64_t>()[0] = total_feature_frames;

    auto encoder_outs = model_->RunEncoder(
        std::move(x), std::move(length),
        model_->GetLanguagePromptId(s->GetOption("language")));
    int32_t samples_per_frame = config_.feat_config.sampling_rate / 100;
    int32_t left_encoder_frames =
        (left_frames * samples_per_frame - start_sample_offset) /
        (samples_per_frame * model_->SubsamplingFactor());
    int64_t encoded_length = encoder_outs[1].GetTensorData<int64_t>()[0];
    valid_center_encoder_frames =
        final_chunk ? encoded_length - left_encoder_frames
                    : std::min<int64_t>(valid_center_encoder_frames,
                                        encoded_length - left_encoder_frames);
    Ort::Value encoder_out = Transpose12(model_->Allocator(), &encoder_outs[0]);
    Ort::Value center_encoder_out = SliceCenterEncoderOut(
        &encoder_out, left_encoder_frames, valid_center_encoder_frames);

    s->GetNumProcessedFrames() += valid_center_frames;

    OnlineStream *streams[1] = {s};
    decoder_->Decode(std::move(center_encoder_out), streams, 1);
  }

 private:
  OnlineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OnlineTransducerNeMoParakeetUnifiedModel> model_;
  std::unique_ptr<OnlineTransducerGreedySearchNeMoParakeetUnifiedDecoder>
      decoder_;
  Endpoint endpoint_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_PARAKEET_UNIFIED_IMPL_H_
