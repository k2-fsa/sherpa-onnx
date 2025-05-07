// sherpa-onnx/csrc/rknn/online-recognizer-ctc-rknn-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_ONLINE_RECOGNIZER_CTC_RKNN_IMPL_H_
#define SHERPA_ONNX_CSRC_RKNN_ONLINE_RECOGNIZER_CTC_RKNN_IMPL_H_

#include <algorithm>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-ctc-decoder.h"
#include "sherpa-onnx/csrc/online-ctc-fst-decoder.h"
#include "sherpa-onnx/csrc/online-ctc-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/rknn/online-stream-rknn.h"
#include "sherpa-onnx/csrc/rknn/online-zipformer-ctc-model-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// defined in ../online-recognizer-ctc-impl.h
OnlineRecognizerResult ConvertCtc(const OnlineCtcDecoderResult &src,
                                  const SymbolTable &sym_table,
                                  float frame_shift_ms,
                                  int32_t subsampling_factor, int32_t segment,
                                  int32_t frames_since_start);

class OnlineRecognizerCtcRknnImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerCtcRknnImpl(const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        model_(
            std::make_unique<OnlineZipformerCtcModelRknn>(config.model_config)),
        endpoint_(config_.endpoint_config) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      /// assuming tokens_buf and tokens are guaranteed not being both empty
      sym_ = SymbolTable(config.model_config.tokens, true);
    }

    InitDecoder();
  }

  template <typename Manager>
  explicit OnlineRecognizerCtcRknnImpl(Manager *mgr,
                                       const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        model_(std::make_unique<OnlineZipformerCtcModelRknn>(
            mgr, config_.model_config)),
        sym_(mgr, config_.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    InitDecoder();
  }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStreamRknn>(config_.feat_config);
    stream->SetZipformerEncoderStates(model_->GetInitStates());
    stream->SetFasterDecoder(decoder_->CreateFasterDecoder());
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(reinterpret_cast<OnlineStreamRknn *>(ss[i]));
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    OnlineCtcDecoderResult decoder_result = s->GetCtcResult();

    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    auto r =
        ConvertCtc(decoder_result, sym_, frame_shift_ms, subsampling_factor,
                   s->GetCurrentSegment(), s->GetNumFramesSinceStart());
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

    // subsampling factor is 4
    int32_t trailing_silence_frames = s->GetCtcResult().num_trailing_blanks * 4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    // segment is incremented only when the last
    // result is not empty
    const auto &r = s->GetCtcResult();
    if (!r.tokens.empty()) {
      s->GetCurrentSegment() += 1;
    }

    // clear result
    s->SetCtcResult({});

    // clear states
    reinterpret_cast<OnlineStreamRknn *>(s)->SetZipformerEncoderStates(
        model_->GetInitStates());

    s->GetFasterDecoderProcessedFrames() = 0;

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

 private:
  void InitDecoder() {
    if (!sym_.Contains("<blk>") && !sym_.Contains("<eps>") &&
        !sym_.Contains("<blank>")) {
      SHERPA_ONNX_LOGE(
          "We expect that tokens.txt contains "
          "the symbol <blk> or <eps> or <blank> and its ID.");
      exit(-1);
    }

    int32_t blank_id = 0;
    if (sym_.Contains("<blk>")) {
      blank_id = sym_["<blk>"];
    } else if (sym_.Contains("<eps>")) {
      // for tdnn models of the yesno recipe from icefall
      blank_id = sym_["<eps>"];
    } else if (sym_.Contains("<blank>")) {
      // for WeNet CTC models
      blank_id = sym_["<blank>"];
    }

    if (!config_.ctc_fst_decoder_config.graph.empty()) {
      decoder_ = std::make_unique<OnlineCtcFstDecoder>(
          config_.ctc_fst_decoder_config, blank_id);
    } else if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineCtcGreedySearchDecoder>(blank_id);
    } else {
      SHERPA_ONNX_LOGE(
          "Unsupported decoding method: %s for streaming CTC models",
          config_.decoding_method.c_str());
      exit(-1);
    }
  }

  void DecodeStream(OnlineStreamRknn *s) const {
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feat_dim = s->FeatureDim();

    const auto num_processed_frames = s->GetNumProcessedFrames();
    std::vector<float> features =
        s->GetFrames(num_processed_frames, chunk_size);
    s->GetNumProcessedFrames() += chunk_shift;

    auto &states = s->GetZipformerEncoderStates();
    auto p = model_->Run(features, std::move(states));
    states = std::move(p.second);

    std::vector<OnlineCtcDecoderResult> results(1);
    results[0] = std::move(s->GetCtcResult());

    auto attr = model_->GetOutAttr();

    decoder_->Decode(p.first.data(), attr.dims[0], attr.dims[1], attr.dims[2],
                     &results, reinterpret_cast<OnlineStream **>(&s), 1);
    s->SetCtcResult(results[0]);
  }

 private:
  OnlineRecognizerConfig config_;
  std::unique_ptr<OnlineZipformerCtcModelRknn> model_;
  std::unique_ptr<OnlineCtcDecoder> decoder_;
  SymbolTable sym_;
  Endpoint endpoint_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_ONLINE_RECOGNIZER_CTC_RKNN_IMPL_H_
