// sherpa-onnx/csrc/offline-recognizer-transducer-nemo-impl.h
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_

#include <fstream>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-transducer-greedy-search-nemo-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-nemo-model.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

// defined in ./offline-recognizer-transducer-impl.h
OfflineRecognitionResult Convert(const OfflineTransducerDecoderResult &src,
                                 const SymbolTable &sym_table,
                                 int32_t frame_shift_ms,
                                 int32_t subsampling_factor);

class OfflineRecognizerTransducerNeMoImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerTransducerNeMoImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerNeMoModel>(
            config_.model_config)) {
    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      exit(-1);
    }
    PostInit();
  }

  template <typename Manager>
  explicit OfflineRecognizerTransducerNeMoImpl(
      Manager *mgr, const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerNeMoModel>(
            mgr, config_.model_config)) {
    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      exit(-1);
    }

    PostInit();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = ss[0]->FeatureDim();

    std::vector<Ort::Value> features;

    features.reserve(n);

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int64_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      auto f = ss[i]->GetFrames();
      int32_t num_frames = f.size() / feat_dim;

      features_length_vec[i] = num_frames;
      features_vec[i] = std::move(f);

      std::array<int64_t, 2> shape = {num_frames, feat_dim};

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }

    std::vector<const Ort::Value *> features_pointer(n);
    for (int32_t i = 0; i != n; ++i) {
      features_pointer[i] = &features[i];
    }

    std::array<int64_t, 1> features_length_shape = {n};
    Ort::Value x_length = Ort::Value::CreateTensor(
        memory_info, features_length_vec.data(), n,
        features_length_shape.data(), features_length_shape.size());

    Ort::Value x = PadSequence(model_->Allocator(), features_pointer, 0);

    auto t = model_->RunEncoder(std::move(x), std::move(x_length));
    // t[0] encoder_out, float tensor, (batch_size, dim, T)
    // t[1] encoder_out_length, int64 tensor, (batch_size,)

    Ort::Value encoder_out = Transpose12(model_->Allocator(), &t[0]);

    auto results = decoder_->Decode(std::move(encoder_out), std::move(t[1]));

    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_, frame_shift_ms,
                       model_->SubsamplingFactor());
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));

      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void PostInit() {
    int32_t feat_dim = model_->FeatureDim();

    if (feat_dim > 0) {
      config_.feat_config.feature_dim = feat_dim;
    }

    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    config_.feat_config.dither = 0;

    if (model_->IsGigaAM()) {
      config_.feat_config.low_freq = 0;
      config_.feat_config.high_freq = 8000;
      config_.feat_config.remove_dc_offset = false;
      config_.feat_config.preemph_coeff = 0;
      config_.feat_config.window_type = "hann";
      config_.feat_config.feature_dim = 64;

      // see
      // https://github.com/salute-developers/GigaAM/blob/main/gigaam/preprocess.py#L68
      //
      // GigaAM uses n_fft 400
      config_.feat_config.round_to_power_of_two = false;
    } else {
      config_.feat_config.low_freq = 0;
      // config_.feat_config.high_freq = 8000;
      config_.feat_config.is_librosa = true;
      config_.feat_config.remove_dc_offset = false;
      // config_.feat_config.window_type = "hann";
    }

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
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineTransducerNeMoModel> model_;
  std::unique_ptr<OfflineTransducerDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_
