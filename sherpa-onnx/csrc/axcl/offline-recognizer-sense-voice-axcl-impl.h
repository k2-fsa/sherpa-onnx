// sherpa-onnx/csrc/axcl/offline-recognizer-sense-voice-axcl-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_OFFLINE_RECOGNIZER_SENSE_VOICE_AXCL_IMPL_H_
#define SHERPA_ONNX_CSRC_AXCL_OFFLINE_RECOGNIZER_SENSE_VOICE_AXCL_IMPL_H_

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/axcl/offline-sense-voice-model-axcl.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/rknn/offline-ctc-greedy-search-decoder-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// defined in ../online-recognizer-sense-voice-impl.h
OfflineRecognitionResult ConvertSenseVoiceResult(
    const OfflineCtcDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor);

class OfflineRecognizerSenseVoiceAxclImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerSenseVoiceAxclImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(
            std::make_unique<OfflineSenseVoiceModelAxcl>(config.model_config)) {
    const auto &meta_data = model_->GetModelMetadata();
    if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineCtcGreedySearchDecoderRknn>(
          meta_data.blank_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    InitFeatConfig();
  }

  template <typename Manager>
  OfflineRecognizerSenseVoiceAxclImpl(Manager *mgr,
                                      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineSenseVoiceModelAxcl>(
            mgr, config.model_config)) {
    const auto &meta_data = model_->GetModelMetadata();
    if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineCtcGreedySearchDecoderRknn>(
          meta_data.blank_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    InitFeatConfig();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeOneStream(ss[i]);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void InitFeatConfig() {
    const auto &meta_data = model_->GetModelMetadata();

    config_.feat_config.normalize_samples = meta_data.normalize_samples;
    config_.feat_config.window_type = "hamming";
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
  }

  void DecodeOneStream(OfflineStream *s) const {
    const auto &meta_data = model_->GetModelMetadata();

    std::vector<float> f = s->GetFrames();

    int32_t language = 0;
    if (config_.model_config.sense_voice.language.empty()) {
      language = 0;
    } else if (meta_data.lang2id.count(
                   config_.model_config.sense_voice.language)) {
      language =
          meta_data.lang2id.at(config_.model_config.sense_voice.language);
    } else {
      SHERPA_ONNX_LOGE("Unknown language: %s. Use 0 instead.",
                       config_.model_config.sense_voice.language.c_str());
    }

    int32_t text_norm = config_.model_config.sense_voice.use_itn
                            ? meta_data.with_itn_id
                            : meta_data.without_itn_id;

    std::vector<float> logits = model_->Run(std::move(f), language, text_norm);
    int32_t num_out_frames = logits.size() / meta_data.vocab_size;

    auto result =
        decoder_->Decode(logits.data(), num_out_frames, meta_data.vocab_size);

    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = meta_data.window_shift;
    auto r = ConvertSenseVoiceResult(result, symbol_table_, frame_shift_ms,
                                     subsampling_factor);

    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    s->SetResult(r);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineSenseVoiceModelAxcl> model_;
  std::unique_ptr<OfflineCtcGreedySearchDecoderRknn> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_OFFLINE_RECOGNIZER_SENSE_VOICE_AXCL_IMPL_H_
