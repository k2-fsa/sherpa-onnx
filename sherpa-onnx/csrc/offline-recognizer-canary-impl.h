// sherpa-onnx/csrc/offline-recognizer-canary-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CANARY_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CANARY_IMPL_H_

#include <algorithm>
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
    auto enc_out = RunEncoder(s);

    if (enc_out.empty()) {
      OfflineRecognitionResult empty_result;
      s->SetResult(empty_result);
      return;
    }

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

    auto [max_token_id, confidence] = GetMaxTokenIdWithConfidence(&logits);

    int32_t eos = symbol_table_["<|endoftext|>"];

    int32_t num_feature_frames =
        enc_states.GetTensorTypeAndShapeInfo().GetShape()[1] *
        meta.subsampling_factor;

    std::vector<int32_t> tokens = {max_token_id};
    std::vector<float> token_log_probs = {confidence};

    // Assume 30 tokens per second. It is to avoid the following for loop
    // running indefinitely.
    int32_t num_tokens =
        static_cast<int32_t>(num_feature_frames / 100.0 * 30) + 1;

    if (max_token_id == eos) {
      // First token is EOS, stop immediately
    }

    // Start from decoder_input.size() + 1 for proper position tracking
    int32_t tokens_generated = 0;
    for (int32_t i = decoder_input.size() + 1;
         i <= decoder_input.size() + num_tokens; ++i) {
      if (tokens.back() == eos) {
        break;
      }

      std::tie(logits, decoder_states) =
          RunDecoder(tokens.back(), i, std::move(decoder_states),
                     View(&enc_states), View(&enc_mask));

      auto [next_token_id, next_confidence] =
          GetMaxTokenIdWithConfidence(&logits);

      tokens.push_back(next_token_id);
      token_log_probs.push_back(next_confidence);
      tokens_generated++;
    }

    // remove the last eos token and its confidence
    if (!tokens.empty() && tokens.back() == eos) {
      tokens.pop_back();
      token_log_probs.pop_back();
    }

    auto r = Convert(tokens, token_log_probs);

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
  OfflineRecognitionResult Convert(
      const std::vector<int32_t> &tokens,
      const std::vector<float> &token_log_probs) const {
    OfflineRecognitionResult r;
    r.tokens.reserve(tokens.size());
    r.token_probs = token_log_probs;

    std::string text;
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
      int32_t token_id = tokens[idx];

      if (!symbol_table_.Contains(token_id)) {
        continue;
      }

      const auto &s = symbol_table_[token_id];
      text += s;
      r.tokens.push_back(s);
    }

    r.text = std::move(text);

    return r;
  }

  std::pair<int32_t, float> GetMaxTokenIdWithConfidence(
      Ort::Value *logits) const {
    // logits is of shape (1, 1, vocab_size)
    auto meta = model_->GetModelMetadata();
    const float *p_logits = logits->GetTensorData<float>();

    // Find max for numerical stability
    float max_logit = *std::max_element(p_logits, p_logits + meta.vocab_size);

    // Compute log_softmax
    float sum_exp = 0.0f;
    for (int32_t i = 0; i < meta.vocab_size; ++i) {
      sum_exp += std::exp(p_logits[i] - max_logit);
    }
    float log_sum = max_logit + std::log(sum_exp);

    // Find the max token and its log probability
    int32_t max_token_id = 0;
    float max_log_prob = p_logits[0] - log_sum;

    for (int32_t i = 1; i < meta.vocab_size; ++i) {
      float log_prob = p_logits[i] - log_sum;
      if (log_prob > max_log_prob) {
        max_log_prob = log_prob;
        max_token_id = i;
      }
    }

    return {max_token_id, max_log_prob};
  }

  std::vector<Ort::Value> RunEncoder(OfflineStream *s) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = config_.feat_config.feature_dim;
    std::vector<float> f = s->GetFrames();

    if (f.empty()) {
      return {};
    }

    int32_t num_frames = f.size() / feat_dim;

    std::array<int64_t, 3> shape = {1, num_frames, feat_dim};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    int64_t x_length_scalar = num_frames;
    std::array<int64_t, 1> x_length_shape = {1};
    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &x_length_scalar, 1,
                                 x_length_shape.data(), x_length_shape.size());

    auto result = model_->ForwardEncoder(std::move(x), std::move(x_length));

    return result;
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

    auto result = model_->ForwardDecoder(
        std::move(decoder_input), std::move(decoder_states),
        std::move(enc_states), std::move(enc_mask));

    return result;
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
      SHERPA_ONNX_LOGE("symbol_table has %d symbols but vocab_size is %d",
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
