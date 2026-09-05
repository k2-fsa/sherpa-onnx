// sherpa-onnx/csrc/offline-recognizer-canary-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CANARY_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CANARY_IMPL_H_

#include <algorithm>
#include <cmath>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-canary-model.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

// The greedy decoder in this file treats eos as "end of transcript". For
// audio that contains signal, eos can win the argmax at the very first
// generated position only as a numerical near-tie (a 0.1% amplitude change
// of the input flips the result; see
// https://github.com/k2-fsa/sherpa-onnx/issues/3919). Once eos has been
// generated, the decoder keeps emitting eos, so the whole transcript would
// be lost and the result would silently be an empty string.
//
// To avoid that, eos is never selected as the first token of an utterance
// that contains signal. For silence, eos is the model's correct
// "nothing was said" answer, so it is kept there.
//
// Silence is detected on the normalized fbank features returned by
// OfflineStream::GetFrames(). Those features are scale invariant, so the
// recording level does not matter. For
// sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8, digital silence gives
// a max |feature| of about 0.09 while any audio with content stays above 4,
// regardless of its level.
inline constexpr float kCanarySilenceFeatureAbsMax = 1.0f;

inline bool CanaryHasSignal(const float *features, int32_t n) {
  float abs_max = 0;
  for (int32_t i = 0; i != n; ++i) {
    abs_max = std::max(abs_max, std::abs(features[i]));
  }

  return abs_max > kCanarySilenceFeatureAbsMax;
}

inline int32_t SelectCanaryFirstToken(const float *logits, int32_t vocab_size,
                                      int32_t eos_id, bool has_signal) {
  int32_t first = static_cast<int32_t>(
      std::distance(logits, std::max_element(logits, logits + vocab_size)));

  if (first != eos_id || !has_signal) {
    return first;
  }

  SHERPA_ONNX_LOGE(
      "The first generated token is <|endoftext|> for audio that contains "
      "signal. Using the best non-eos token instead, otherwise the "
      "transcript would be empty (see issue #3919)");

  int32_t best = eos_id;
  float best_logit = 0;
  for (int32_t i = 0; i != vocab_size; ++i) {
    if (i == eos_id) {
      continue;
    }

    if (best == eos_id || logits[i] > best_logit) {
      best = i;
      best_logit = logits[i];
    }
  }

  return best;
}

class OfflineRecognizerCanaryImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerCanaryImpl(const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineCanaryModel>(config_.model_config)) {
    PostInit();
  }

  template <typename Manager>
  explicit OfflineRecognizerCanaryImpl(Manager *mgr,
                                       const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(
            std::make_unique<OfflineCanaryModel>(mgr, config_.model_config)) {
    PostInit();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  void DecodeStream(OfflineStream *s) const {
    auto meta = model_->GetModelMetadata();
    std::vector<float> frames = s->GetFrames();
    auto enc_out = RunEncoder(frames);
    Ort::Value enc_states = std::move(enc_out[0]);
    Ort::Value enc_mask = std::move(enc_out[2]);
    // enc_out[1] is discarded
    std::vector<int32_t> decoder_input = GetInitialDecoderInput();
    auto decoder_states = model_->GetInitialDecoderStates();
    Ort::Value logits{nullptr};

    for (int32_t i = 0; i < decoder_input.size(); ++i) {
      std::tie(logits, decoder_states) =
          RunDecoder(decoder_input[i], i, std::move(decoder_states),
                     View(&enc_states), View(&enc_mask));
    }

    int32_t eos = symbol_table_["<|endoftext|>"];

    int32_t num_feature_frames =
        enc_states.GetTensorTypeAndShapeInfo().GetShape()[1] *
        meta.subsampling_factor;

    bool has_signal = CanaryHasSignal(frames.data(), frames.size());

    std::vector<int32_t> tokens = {SelectCanaryFirstToken(
        logits.GetTensorData<float>(), meta.vocab_size, eos, has_signal)};

    // Assume 30 tokens per second. It is to avoid the following for loop
    // running indefinitely.
    int32_t num_tokens =
        static_cast<int32_t>(num_feature_frames / 100.0 * 30) + 1;

    for (int32_t i = 1; i <= num_tokens; ++i) {
      if (tokens.back() == eos) {
        break;
      }

      std::tie(logits, decoder_states) =
          RunDecoder(tokens.back(), i, std::move(decoder_states),
                     View(&enc_states), View(&enc_mask));
      tokens.push_back(GetMaxTokenId(&logits));
    }

    // remove the last eos token
    tokens.pop_back();

    auto r = Convert(tokens);

    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));

    s->SetResult(r);
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

  void SetConfig(const OfflineRecognizerConfig &config) override {
    config_.model_config.canary.src_lang = config.model_config.canary.src_lang;
    config_.model_config.canary.tgt_lang = config.model_config.canary.tgt_lang;
    config_.model_config.canary.use_pnc = config.model_config.canary.use_pnc;

    // we don't change the config_ in the base class
  }

 private:
  OfflineRecognitionResult Convert(const std::vector<int32_t> &tokens) const {
    OfflineRecognitionResult r;
    r.tokens.reserve(tokens.size());

    std::string text;
    for (auto i : tokens) {
      if (!symbol_table_.Contains(i)) {
        continue;
      }

      const auto &s = symbol_table_[i];
      text += s;
      r.tokens.push_back(s);
    }

    r.text = std::move(text);

    return r;
  }

  int32_t GetMaxTokenId(Ort::Value *logits) const {
    // logits is of shape (1, 1, vocab_size)
    auto meta = model_->GetModelMetadata();
    const float *p_logits = logits->GetTensorData<float>();

    int32_t max_token_id = static_cast<int32_t>(std::distance(
        p_logits, std::max_element(p_logits, p_logits + meta.vocab_size)));

    return max_token_id;
  }

  std::vector<Ort::Value> RunEncoder(std::vector<float> &frames) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = config_.feat_config.feature_dim;

    int32_t num_frames = frames.size() / feat_dim;

    std::array<int64_t, 3> shape = {1, num_frames, feat_dim};

    Ort::Value x = Ort::Value::CreateTensor(
        memory_info, frames.data(), frames.size(), shape.data(), shape.size());

    int64_t x_length_scalar = num_frames;
    std::array<int64_t, 1> x_length_shape = {1};
    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &x_length_scalar, 1,
                                 x_length_shape.data(), x_length_shape.size());
    return model_->ForwardEncoder(std::move(x), std::move(x_length));
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> RunDecoder(
      int32_t token, int32_t pos, std::vector<Ort::Value> decoder_states,
      Ort::Value enc_states, Ort::Value enc_mask) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> shape = {1, 2};
    std::array<int32_t, 2> _decoder_input = {token, pos};

    Ort::Value decoder_input = Ort::Value::CreateTensor(
        memory_info, _decoder_input.data(), _decoder_input.size(), shape.data(),
        shape.size());

    return model_->ForwardDecoder(std::move(decoder_input),
                                  std::move(decoder_states),
                                  std::move(enc_states), std::move(enc_mask));
  }

  // see
  // https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/canary/test_180m_flash.py#L242
  std::vector<int32_t> GetInitialDecoderInput() const {
    auto canary_config = config_.model_config.canary;
    const auto &meta = model_->GetModelMetadata();

    std::vector<int32_t> decoder_input(9);
    decoder_input[0] = symbol_table_["<|startofcontext|>"];
    decoder_input[1] = symbol_table_["<|startoftranscript|>"];
    decoder_input[2] = symbol_table_["<|emo:undefined|>"];

    if (canary_config.src_lang.empty() ||
        !meta.lang2id.count(canary_config.src_lang)) {
      decoder_input[3] = meta.lang2id.at("en");
    } else {
      decoder_input[3] = meta.lang2id.at(canary_config.src_lang);
    }

    if (canary_config.tgt_lang.empty() ||
        !meta.lang2id.count(canary_config.tgt_lang)) {
      decoder_input[4] = meta.lang2id.at("en");
    } else {
      decoder_input[4] = meta.lang2id.at(canary_config.tgt_lang);
    }

    if (canary_config.use_pnc) {
      decoder_input[5] = symbol_table_["<|pnc|>"];
    } else {
      decoder_input[5] = symbol_table_["<|nopnc|>"];
    }

    decoder_input[6] = symbol_table_["<|noitn|>"];
    decoder_input[7] = symbol_table_["<|notimestamp|>"];
    decoder_input[8] = symbol_table_["<|nodiarize|>"];

    return decoder_input;
  }

 private:
  void PostInit() {
    auto &meta = model_->GetModelMetadata();
    config_.feat_config.feature_dim = meta.feat_dim;

    config_.feat_config.nemo_normalize_type = meta.normalize_type;

    config_.feat_config.dither = 0;
    config_.feat_config.remove_dc_offset = false;
    config_.feat_config.low_freq = 0;
    config_.feat_config.window_type = "hann";
    config_.feat_config.is_librosa = true;

    meta.lang2id["en"] = symbol_table_["<|en|>"];
    meta.lang2id["es"] = symbol_table_["<|es|>"];
    meta.lang2id["de"] = symbol_table_["<|de|>"];
    meta.lang2id["fr"] = symbol_table_["<|fr|>"];

    if (symbol_table_.NumSymbols() != meta.vocab_size) {
      SHERPA_ONNX_LOGE("number of lines in tokens.txt %d != %d (vocab_size)",
                       symbol_table_.NumSymbols(), meta.vocab_size);
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineCanaryModel> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CANARY_IMPL_H_
