// sherpa-onnx/csrc/offline-recognizer-whisper-tpl-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_TPL_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_TPL_IMPL_H_

#include <memory>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

template <typename WhisperModel>
class OfflineRecognizerWhisperTplImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerWhisperTplImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<WhisperModel>(config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerWhisperTplImpl(Manager *mgr,
                                  const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<WhisperModel>(mgr, config.model_config)) {
    Init();
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
  void Init() {
    // tokens.txt from whisper is base64 encoded, so we need to decode it
    symbol_table_.ApplyBase64Decode();

    if (config_.decoding_method == "greedy_search") {
      SHERPA_ONNX_LOGE("use greedy_search");
    } else {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported at present for whisper. Given '%s'",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void DecodeStream(OfflineStream *s) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();
    int32_t num_frames = f.size() / feat_dim;

    NormalizeWhisperFeatures(f.data(), num_frames, feat_dim);

    auto r = model_->Run(std::move(f));
    auto res = Convert(r, symbol_table_);

    s->SetResult(res);
  }

  OfflineRecognitionResult Convert(const OfflineWhisperDecoderResult &src,
                                   const SymbolTable &sym_table) const {
    OfflineRecognitionResult r;
    r.tokens.reserve(src.tokens.size());

    std::string text;
    for (auto i : src.tokens) {
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

    return r;
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<WhisperModel> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_TPL_IMPL_H_
