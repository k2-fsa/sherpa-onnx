// sherpa-onnx/csrc/offline-recognizer-transducer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-model.h"
#include "sherpa-onnx/csrc/offline-transducer-modified-beam-search-decoder.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

static OfflineRecognitionResult Convert(
    const OfflineTransducerDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    text.append(sym);

    r.tokens.push_back(std::move(sym));
  }
  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  return r;
}

class OfflineRecognizerTransducerImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerTransducerImpl(
      const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerModel>(config_.model_config)) {
    if (config_.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineTransducerGreedySearchDecoder>(model_.get());
    } else if (config_.decoding_method == "modified_beam_search") {
      if (!config_.lm_config.model.empty()) {
        lm_ = OfflineLM::Create(config.lm_config);
      }

      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchDecoder>(
          model_.get(), lm_.get(), config_.max_active_paths,
          config_.lm_config.scale);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      exit(-1);
    }
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

    Ort::Value x = PadSequence(model_->Allocator(), features_pointer,
                               -23.025850929940457f);

    auto t = model_->RunEncoder(std::move(x), std::move(x_length));
    auto results = decoder_->Decode(std::move(t.first), std::move(t.second));

    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_, frame_shift_ms,
                       model_->SubsamplingFactor());

      ss[i]->SetResult(r);
    }
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineTransducerModel> model_;
  std::unique_ptr<OfflineTransducerDecoder> decoder_;
  std::unique_ptr<OfflineLM> lm_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
