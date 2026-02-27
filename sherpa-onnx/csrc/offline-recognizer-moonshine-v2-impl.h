// sherpa-onnx/csrc/offline-recognizer-moonshine-v2-impl.h
//
// Copyright (c)  2024-2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_MOONSHINE_V2_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_MOONSHINE_V2_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-moonshine-decoder.h"
#include "sherpa-onnx/csrc/offline-moonshine-model-v2.h"
#include "sherpa-onnx/csrc/offline-moonshine-v2-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

// defined in ./offline-recognizer-moonshine-impl.h
OfflineRecognitionResult Convert(const OfflineMoonshineDecoderResult &src,
                                 const SymbolTable &sym_table);

class OfflineRecognizerMoonshineV2Impl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerMoonshineV2Impl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineMoonshineModelV2>(config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerMoonshineV2Impl(Manager *mgr,
                                   const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineMoonshineModelV2>(mgr,
                                                         config.model_config)) {
    Init();
  }

  void Init() {
    // tokens.txt from whisper is base64 encoded, so we need to decode it
    // See also ../../scripts/moonshine/v2/generate_tokens.py
    symbol_table_.ApplyBase64Decode();

    if (config_.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineMoonshineV2GreedySearchDecoder>(model_.get());
    } else {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported at present for moonshine. Given %s",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    MoonshineTag tag;
    return std::make_unique<OfflineStream>(tag);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    // batch decoding is not implemented yet
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeStream(OfflineStream *s) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<float> audio = s->GetFrames();

    try {
      std::array<int64_t, 2> shape{1, static_cast<int64_t>(audio.size())};

      Ort::Value audio_tensor = Ort::Value::CreateTensor(
          memory_info, audio.data(), audio.size(), shape.data(), shape.size());

      Ort::Value encoder_out = model_->ForwardEncoder(std::move(audio_tensor));

      auto results = decoder_->Decode(std::move(encoder_out));

      auto r = Convert(results[0], symbol_table_);
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));
      s->SetResult(r);
    } catch (const Ort::Exception &ex) {
      SHERPA_ONNX_LOGE(
          "\n\nCaught exception:\n\n%s\n\nReturn an empty result. Number of "
          "audio samples: %d",
          ex.what(), static_cast<int32_t>(audio.size()));
      return;
    }
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineMoonshineModelV2> model_;
  std::unique_ptr<OfflineMoonshineDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_MOONSHINE_V2_IMPL_H_
