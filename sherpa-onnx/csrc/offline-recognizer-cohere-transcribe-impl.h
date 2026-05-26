// sherpa-onnx/csrc/offline-recognizer-cohere-transcribe-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_COHERE_TRANSCRIBE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_COHERE_TRANSCRIBE_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-cohere-transcribe-decoder.h"
#include "sherpa-onnx/csrc/offline-cohere-transcribe-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-cohere-transcribe-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

class OfflineRecognizerCohereTranscribeImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerCohereTranscribeImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineCohereTranscribeModel>(
            config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerCohereTranscribeImpl(Manager *mgr,
                                        const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineCohereTranscribeModel>(
            mgr, config.model_config)) {
    Init();
  }

  void Init() {
    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineCohereTranscribeGreedySearchDecoder>(
          model_.get());
    } else {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported at present for Cohere Transcribe. "
          "Given %s",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    config_.feat_config.feature_dim = 128;
    config_.feat_config.nemo_normalize_type = "per_feature";
    config_.feat_config.low_freq = 0;
    config_.feat_config.remove_dc_offset = false;
    config_.feat_config.window_type = "hann";
    config_.feat_config.is_librosa = true;
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
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
    auto language = s->GetOption("language");

    if (language.empty()) {
      language = config_.model_config.cohere_transcribe.language;
    }

    if (language.empty()) {
      SHERPA_ONNX_LOGE("Please specify a language for Cohere Transcribe");
      return;
    }
    if (!IsValidCohereTranscribeLanguage(language)) {
      SHERPA_ONNX_LOGE(
          "Invalid language: '%s'. "
          "Supported values: ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, "
          "vi, zh",
          language.c_str());
      return;
    }

    bool use_itn = s->GetOptionInt(
        "use_itn", config_.model_config.cohere_transcribe.use_itn);

    bool use_punct = s->GetOptionInt(
        "use_punct", config_.model_config.cohere_transcribe.use_punct);

    std::vector<std::string> prompt_str;
    prompt_str.reserve(9);
    prompt_str.push_back("<|startofcontext|>");
    prompt_str.push_back("<|startoftranscript|>");
    prompt_str.push_back("<|emo:undefined|>");
    prompt_str.push_back(std::string("<|") + language + "|>");
    prompt_str.push_back(std::string("<|") + language + "|>");
    if (use_punct) {
      prompt_str.push_back("<|pnc|>");
    } else {
      prompt_str.push_back("<|nopnc|>");
    }

    if (use_itn) {
      prompt_str.push_back("<|itn|>");
    } else {
      prompt_str.push_back("<|noitn|>");
    }
    prompt_str.push_back("<|notimestamp|>");
    prompt_str.push_back("<|nodiarize|>");

    std::vector<int64_t> prompt_ids;
    prompt_ids.reserve(prompt_str.size());
    for (const auto &str : prompt_str) {
      prompt_ids.push_back(symbol_table_[str]);
    }

    int64_t eos = symbol_table_["<|endoftext|>"];

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();
    int32_t num_frames = f.size() / feat_dim;

    f = Transpose(f.data(), num_frames, feat_dim);
    // now f is (1, feat_dim, num_frames)
    std::array<int64_t, 3> shape{1, feat_dim, num_frames};
    Ort::Value mel = Ort::Value::CreateTensor(memory_info, f.data(), f.size(),
                                              shape.data(), shape.size());

    try {
      auto cross_kv = model_->ForwardEncoder(std::move(mel));

      auto results = decoder_->Decode(std::move(cross_kv.first),
                                      std::move(cross_kv.second), prompt_ids,
                                      eos, num_frames);

      auto r = Convert(results[0], symbol_table_);
      s->SetResult(r);
    } catch (const Ort::Exception &ex) {
      SHERPA_ONNX_LOGE(
          "\n\nCaught exception:\n\n%s\n\nReturn an empty result. Number of "
          "input frames: %d.",
          ex.what(), num_frames);
      return;
    }
  }

 private:
  OfflineRecognitionResult Convert(
      const OfflineCohereTranscribeDecoderResult &src,
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

    return r;
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineCohereTranscribeModel> model_;
  std::unique_ptr<OfflineCohereTranscribeDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_COHERE_TRANSCRIBE_IMPL_H_
