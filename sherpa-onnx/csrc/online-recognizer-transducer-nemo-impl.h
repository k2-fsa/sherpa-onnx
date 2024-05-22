// sherpa-onnx/csrc/online-recognizer-transducer-nemo-impl.h
//
// Copyright (c)  2022-2024  Xiaomi Corporation

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
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

// defined in ./online-recognizer-transducer-impl.h
OnlineRecognizerResult Convert(const OnlineTransducerDecoderResult &src,
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
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OnlineTransducerNeMoModel>(
            config_.model_config)) {
    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      exit(-1);
    }
    PostInit();
  }

#if __ANDROID_API__ >= 9
  explicit OnlineRecognizerTransducerNeMoImpl(
      AAssetManager *mgr, const OnlineRecognizerConfig &config)
      : config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OnlineTransducerNeMoModel>(
            mgr, config_.model_config)) {
    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                      config_.decoding_method.c_str());
      exit(-1);
    }

    PostInit();
  }
#endif

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStream>(config_.feat_config);
    InitOnlineStream(stream.get());
    return stream;
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    int32_t feature_dim = ss[0]->FeatureDim();

    std::vector<OnlineTransducerDecoderResult> results(n);
    std::vector<float> features_vec(n * chunk_size * feature_dim);
    std::vector<std::vector<Ort::Value>> states_vec(n);
    std::vector<int64_t> all_processed_frames(n);

    for (int32_t i = 0; i != n; ++i) {
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

    auto [t, ns] = model_->RunEncoder(std::move(x), std::move(states),
                                   std::move(processed_frames));

    Ort::Value encoder_out = Transpose12(model_->Allocator(), &t[0]);
    
    // defined in online-transducer-greedy-search-nemo-decoder.h
     auto results = decoder_-> Decode(std::move(encoder_out), std::move(t[1]));

    std::vector<std::vector<Ort::Value>> next_states =
        model_->UnStackStates(ns);

    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(results[i]);
      ss[i]->SetNeMoDecoderStates(std::move(next_states[i]));
    }
  }

  void InitOnlineStream(OnlineStream *stream) const {
    auto r = decoder_->GetEmptyResult();

    stream->SetResult(r);
    stream->SetNeMoDecoderStates(model_->GetDecoderInitStates(batch_size_));
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
  std::unique_ptr<OnlineTransducerDecoder> decoder_;

   int32_t batch_size_ = 1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_