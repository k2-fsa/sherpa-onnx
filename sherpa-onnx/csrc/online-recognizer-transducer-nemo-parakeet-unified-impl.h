// sherpa-onnx/csrc/online-recognizer-transducer-nemo-parakeet-unified-impl.h
//
// Copyright (c)  2026  Milan Leonard

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
                               int32_t segment, int32_t frames_since_start);

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

    if (processed + model_->ChunkFeatureFrames() +
            model_->RightFeatureFrames() <=
        ready) {
      return true;
    }

    // Flush the final, possibly short, center chunk with zero-padded right
    // context after InputFinished().
    return ready > processed && s->IsLastFrame(ready - 1);
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = model_->SubsamplingFactor();
    auto r = Convert(s->GetResult(), symbol_table_, frame_shift_ms,
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
    s->SetNeMoDecoderStates(model_->GetDecoderInitStates());
    s->Reset();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      DecodeOneStream(ss[i]);
    }
  }

  void InitOnlineStream(OnlineStream *stream) const {
    stream->SetNeMoDecoderStates(model_->GetDecoderInitStates());
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

  std::vector<float> CopyWindowWithPadding(OnlineStream *s) const {
    int32_t feature_dim = model_->FeatureDim();
    int32_t total = model_->TotalFeatureFrames();
    std::vector<float> features(total * feature_dim);

    int32_t processed = s->GetNumProcessedFrames();
    int32_t window_start = processed - model_->LeftFeatureFrames();
    int32_t window_end = window_start + total;
    int32_t read_start = std::max(0, window_start);
    int32_t read_end = std::min(window_end, s->NumFramesReady());

    if (read_end <= read_start) {
      return features;
    }

    std::vector<float> frames = s->GetFrames(read_start, read_end - read_start);
    float *copy_dst =
        features.data() + (read_start - window_start) * feature_dim;
    std::copy(frames.begin(), frames.end(), copy_dst);
    return features;
  }

  Ort::Value SliceCenterEncoderOut(Ort::Value *encoder_out,
                                   int32_t center_frames) const {
    auto shape = encoder_out->GetTensorTypeAndShapeInfo().GetShape();
    int32_t num_rows = static_cast<int32_t>(shape[1]);
    int32_t num_cols = static_cast<int32_t>(shape[2]);
    int32_t start = model_->LeftEncoderFrames();
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
    int32_t valid_center_frames =
        std::min(model_->ChunkFeatureFrames(), ready - processed);

    if (valid_center_frames <= 0) {
      return;
    }

    int32_t feature_dim = model_->FeatureDim();
    int32_t total_feature_frames = model_->TotalFeatureFrames();
    std::vector<float> features = CopyWindowWithPadding(s);

    if (model_->FeatureNormalizationMethod() == "per_feature") {
      NemoNormalizePerFeature(features.data(), total_feature_frames,
                              feature_dim);
    }

    int32_t valid_center_encoder_frames =
        (valid_center_frames + model_->SubsamplingFactor() - 1) /
        model_->SubsamplingFactor();
    valid_center_encoder_frames =
        std::min(valid_center_encoder_frames, model_->ChunkEncoderFrames());

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{1, total_feature_frames, feature_dim};
    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, features.data(), features.size(),
                                 x_shape.data(), x_shape.size());

    std::array<int64_t, 1> length_shape{1};
    Ort::Value length = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), length_shape.data(), length_shape.size());
    length.GetTensorMutableData<int64_t>()[0] = total_feature_frames;

    auto encoder_outs = model_->RunEncoder(std::move(x), std::move(length));
    Ort::Value encoder_out = Transpose12(model_->Allocator(), &encoder_outs[0]);
    Ort::Value center_encoder_out =
        SliceCenterEncoderOut(&encoder_out, valid_center_encoder_frames);

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
