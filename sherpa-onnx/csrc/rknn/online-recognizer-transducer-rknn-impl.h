// sherpa-onnx/csrc/rknn/online-recognizer-transducer-rknn-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_ONLINE_RECOGNIZER_TRANSDUCER_RKNN_IMPL_H_
#define SHERPA_ONNX_CSRC_RKNN_ONLINE_RECOGNIZER_TRANSDUCER_RKNN_IMPL_H_

#include <algorithm>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/rknn/online-stream-rknn.h"
#include "sherpa-onnx/csrc/rknn/online-transducer-greedy-search-decoder-rknn.h"
#include "sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

OnlineRecognizerResult Convert(const OnlineTransducerDecoderResultRknn &src,
                               const SymbolTable &sym_table,
                               float frame_shift_ms, int32_t subsampling_factor,
                               int32_t segment, int32_t frames_since_start) {
  OnlineRecognizerResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.tokens.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];

    text.append(sym);

    if (sym.size() == 1 && (sym[0] < 0x20 || sym[0] > 0x7e)) {
      // for bpe models with byte_fallback
      // (but don't rewrite printable characters 0x20..0x7e,
      //  which collide with standard BPE units)
      std::ostringstream os;
      os << "<0x" << std::hex << std::uppercase
         << (static_cast<int32_t>(sym[0]) & 0xff) << ">";
      sym = os.str();
    }

    r.tokens.push_back(std::move(sym));
  }

  if (sym_table.IsByteBpe()) {
    text = sym_table.DecodeByteBpe(text);
  }

  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  r.segment = segment;
  r.start_time = frames_since_start * frame_shift_ms / 1000.;

  return r;
}

class OnlineRecognizerTransducerRknnImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerTransducerRknnImpl(
      const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        endpoint_(config_.endpoint_config),
        model_(std::make_unique<OnlineZipformerTransducerModelRknn>(
            config.model_config)) {
    decoder_ = std::make_unique<OnlineTransducerGreedySearchDecoderRknn>(
        model_.get(), unk_id_);

    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      /// assuming tokens_buf and tokens are guaranteed not being both empty
      sym_ = SymbolTable(config.model_config.tokens, true);
    }

    if (sym_.Contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }
  }

  template <typename Manager>
  explicit OnlineRecognizerTransducerRknnImpl(
      Manager *mgr, const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        endpoint_(config_.endpoint_config),
        model_(std::make_unique<OnlineZipformerTransducerModelRknn>(mgr,
                                                                    config)) {}

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStreamRknn>(config_.feat_config);
    auto r = decoder_->GetEmptyResult();
    stream->SetZipformerResult(r);
    stream->SetZipformerEncoderStates(model_->GetEncoderInitStates());
    return stream;
  }

  std::unique_ptr<OnlineStream> CreateStream(
      const std::string &hotwords) const override {
    SHERPA_ONNX_LOGE("Hotwords for RKNN is not supported now.");
    return CreateStream();
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  // Warmping up engine with wp: warm_up count and max-batch-size

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeStream(reinterpret_cast<OnlineStreamRknn *>(ss[i]));
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    OnlineTransducerDecoderResultRknn decoder_result =
        reinterpret_cast<OnlineStreamRknn *>(s)->GetZipformerResult();
    decoder_->StripLeadingBlanks(&decoder_result);
    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    auto r = Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor,
                     s->GetCurrentSegment(), s->GetNumFramesSinceStart());
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    return r;
  }

  bool IsEndpoint(OnlineStream *s) const override { return false; }

  void Reset(OnlineStream *s) const override {}

 private:
  void DecodeStream(OnlineStreamRknn *s) const {
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

    auto &r = s->GetZipformerResult();
    decoder_->Decode(std::move(p.first), &r);
  }

 private:
  OnlineRecognizerConfig config_;
  SymbolTable sym_;
  Endpoint endpoint_;
  int32_t unk_id_ = -1;
  std::unique_ptr<OnlineZipformerTransducerModelRknn> model_;
  std::unique_ptr<OnlineTransducerGreedySearchDecoderRknn> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_ONLINE_RECOGNIZER_TRANSDUCER_RKNN_IMPL_H_
