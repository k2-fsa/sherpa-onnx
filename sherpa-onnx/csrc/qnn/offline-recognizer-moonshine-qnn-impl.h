// sherpa-onnx/csrc/qnn/offline-recognizer-moonshine-qnn-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_MOONSHINE_QNN_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_MOONSHINE_QNN_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-moonshine-decoder.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "offline-moonshine-model-qnn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// defined in ./offline-recognizer-moonshine-impl.h
OfflineRecognitionResult Convert(const OfflineMoonshineDecoderResult &src,
                                 const SymbolTable &sym_table);

class OfflineRecognizerMoonshineQnnImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerMoonshineQnnImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineMoonshineModelQnn>(config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerMoonshineQnnImpl(Manager *mgr,
                                    const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineMoonshineModelQnn>(mgr,
                                                          config.model_config)) {
    Init();
  }

  void Init() {
    // tokens.txt is base64 encoded
    symbol_table_.ApplyBase64Decode();
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
    std::vector<float> audio = s->GetFrames();

    auto result = model_->Run(audio);

    auto r = Convert(result, symbol_table_);
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    s->SetResult(r);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineMoonshineModelQnn> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_MOONSHINE_QNN_IMPL_H_
