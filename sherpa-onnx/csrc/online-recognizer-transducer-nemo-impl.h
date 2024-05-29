// sherpa-onnx/csrc/online-recognizer-transducer-nemo-impl.h
//
// Copyright (c)  2022-2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_

#include <fstream>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

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
// static may or may not be here? TODDOs
static OnlineRecognizerResult Convert(const OnlineTransducerDecoderResult &src,
                              const SymbolTable &sym_table,
                              float frame_shift_ms,
                              int32_t subsampling_factor,
                              int32_t segment,
                              int32_t frames_since_start);

class OnlineRecognizerTransducerNeMoImpl : public OnlineRecognizerImpl {
  public:
  explicit OnlineRecognizerTransducerNeMoImpl(
      const OnlineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config.model_config.tokens),
        endpoint_(config_.endpoint_config),
        model_(std::make_unique<OnlineTransducerNeMoModel>(
            config.model_config)) {
    if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }
    PostInit();
  }

#if __ANDROID_API__ >= 9
  explicit OnlineRecognizerTransducerNeMoImpl(
      AAssetManager *mgr, const OnlineRecognizerConfig &config)
      : config_(config),
        symbol_table_(mgr, config.model_config.tokens),
        endpoint_(mgrconfig_.endpoint_config),
        model_(std::make_unique<OnlineTransducerNeMoModel>(
            mgr, config.model_config)) {
    if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                      config.decoding_method.c_str());
      exit(-1);
    }

    PostInit();
  }
#endif

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStream>(config_.feat_config);
    stream->SetStates(model_->GetInitStates());
    InitOnlineStream(stream.get());
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    OnlineTransducerDecoderResult decoder_result = s->GetResult();
    decoder_->StripLeadingBlanks(&decoder_result);

    // TODO(fangjun): Remember to change these constants if needed
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 8;
    return Convert(decoder_result, symbol_table_, frame_shift_ms, subsampling_factor,
                   s->GetCurrentSegment(), s->GetNumFramesSinceStart());
  }

  bool IsEndpoint(OnlineStream *s) const override {
    if (!config_.enable_endpoint) {
      return false;
    }

    int32_t num_processed_frames = s->GetNumProcessedFrames();

    // frame shift is 10 milliseconds
    float frame_shift_in_seconds = 0.01;

    // subsampling factor is 8
    int32_t trailing_silence_frames = s->GetResult().num_trailing_blanks * 8;

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
    
    s->SetResult(r);
    s->GetResult().decoder_out = std::move(decoder_out);

    // Note: We only update counters. The underlying audio samples
    // are not discarded.
    s->Reset();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feature_dim = ss[0]->FeatureDim();

    std::vector<OnlineTransducerDecoderResult> result(n);
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

      result[i] = std::move(ss[i]->GetResult());
      encoder_states[i] = std::move(ss[i]->GetStates());
      
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{n, chunk_size, feature_dim};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, features_vec.data(),
                                            features_vec.size(), x_shape.data(),
                                            x_shape.size());

    // Batch size is 1
    auto states = std::move(encoder_states[0]);
    int32_t num_states = states.size(); // num_states = 3
    auto t = model_->RunEncoder(std::move(x), std::move(states));
    // t[0] encoder_out, float tensor, (batch_size, dim, T)
    // t[1] next states
    
    std::vector<Ort::Value> out_states;
    out_states.reserve(num_states);
    
    for (int32_t k = 1; k != num_states + 1; ++k) {
      out_states.push_back(std::move(t[k]));
    }

    Ort::Value encoder_out = Transpose12(model_->Allocator(), &t[0]);
    
    // defined in online-transducer-greedy-search-nemo-decoder.h
    // get intial states of decoder.
    std::vector<Ort::Value> &decoder_states = ss[0]->GetNeMoDecoderStates();
    
    // Subsequent decoder states (for each chunks) are updated inside the Decode method.
    // This returns the decoder state from the LAST chunk. We probably dont need it. So we can discard it.
    decoder_states = decoder_->Decode(std::move(encoder_out), 
                                      std::move(decoder_states),
                                      &result, ss, n);

    ss[0]->SetResult(result[0]);

    ss[0]->SetStates(std::move(out_states));
  }

  void InitOnlineStream(OnlineStream *stream) const {
    auto r = decoder_->GetEmptyResult();

    stream->SetResult(r);
    stream->SetNeMoDecoderStates(model_->GetDecoderInitStates(1));
  }

 private:
  void PostInit() {
    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    config_.feat_config.low_freq = 0;
    // config_.feat_config.high_freq = 8000;
    config_.feat_config.is_librosa = true;
    config_.feat_config.remove_dc_offset = false;
    // config_.feat_config.window_type = "hann";
    config_.feat_config.dither = 0;
    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    int32_t vocab_size = model_->VocabSize();

    // check the blank ID
    if (!symbol_table_.Contains("<blk>")) {
      SHERPA_ONNX_LOGE("tokens.txt does not include the blank token <blk>");
      exit(-1);
    }

    if (symbol_table_["<blk>"] != vocab_size - 1) {
      SHERPA_ONNX_LOGE("<blk> is not the last token!");
      exit(-1);
    }

    if (symbol_table_.NumSymbols() != vocab_size) {
      SHERPA_ONNX_LOGE("number of lines in tokens.txt %d != %d (vocab_size)",
                       symbol_table_.NumSymbols(), vocab_size);
      exit(-1);
    }

  }

 private:
  OnlineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OnlineTransducerNeMoModel> model_;
  std::unique_ptr<OnlineTransducerGreedySearchNeMoDecoder> decoder_;
  Endpoint endpoint_;

};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_