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
    SHERPA_ONNX_LOGE("CANARY DecodeStreams: Processing %d streams", n);
    for (int32_t i = 0; i < n; ++i) {
      SHERPA_ONNX_LOGE("CANARY DecodeStreams: Decoding stream %d/%d", i+1, n);
      DecodeStream(ss[i]);
    }
    SHERPA_ONNX_LOGE("CANARY DecodeStreams: Finished all %d streams", n);
  }

  void DecodeStream(OfflineStream *s) const {
    SHERPA_ONNX_LOGE("CANARY DecodeStream: ========== START ==========");
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Getting model metadata");
    
    auto meta = model_->GetModelMetadata();
    SHERPA_ONNX_LOGE("CANARY DecodeStream: vocab_size=%d, subsampling_factor=%d", 
                     meta.vocab_size, meta.subsampling_factor);
    
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Running encoder");
    auto enc_out = RunEncoder(s);
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Encoder returned %zu outputs", enc_out.size());
    
    if (enc_out.empty()) {
      SHERPA_ONNX_LOGE("CANARY DecodeStream: ERROR - Encoder returned empty!");
      OfflineRecognitionResult empty_result;
      s->SetResult(empty_result);
      return;
    }
    
    Ort::Value enc_states = std::move(enc_out[0]);
    Ort::Value enc_mask = std::move(enc_out[2]);
    
    auto enc_shape = enc_states.GetTensorTypeAndShapeInfo().GetShape();
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Encoder states shape: [%ld, %ld, %ld]", 
                     enc_shape[0], enc_shape[1], enc_shape[2]);
    
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Getting initial decoder input");
    std::vector<int32_t> decoder_input = GetInitialDecoderInput();
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Initial decoder input has %zu tokens", 
                     decoder_input.size());
    
    for (size_t i = 0; i < decoder_input.size(); ++i) {
      SHERPA_ONNX_LOGE("CANARY DecodeStream: Initial token[%zu] = %d", i, decoder_input[i]);
    }
    
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Getting initial decoder states");
    auto decoder_states = model_->GetInitialDecoderStates();
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Got %zu decoder states", decoder_states.size());
    
    Ort::Value logits{nullptr};

    // Process initial decoder input tokens
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Processing initial %zu tokens", decoder_input.size());
    for (int32_t i = 0; i < decoder_input.size(); ++i) {
      SHERPA_ONNX_LOGE("CANARY DecodeStream: Processing initial token %d at position %d", 
                       decoder_input[i], i);
      std::tie(logits, decoder_states) =
          RunDecoder(decoder_input[i], i, std::move(decoder_states),
                     View(&enc_states), View(&enc_mask));
      SHERPA_ONNX_LOGE("CANARY DecodeStream: Initial token %d processed", i);
    }

    SHERPA_ONNX_LOGE("CANARY DecodeStream: Getting first real token after initial setup");
    auto [max_token_id, confidence] = GetMaxTokenIdWithConfidence(&logits);
    SHERPA_ONNX_LOGE("CANARY DecodeStream: First token=%d, confidence=%.4f", 
                     max_token_id, confidence);
    
    int32_t eos = symbol_table_["<|endoftext|>"];
    SHERPA_ONNX_LOGE("CANARY DecodeStream: EOS token ID = %d", eos);

    int32_t num_feature_frames =
        enc_states.GetTensorTypeAndShapeInfo().GetShape()[1] *
        meta.subsampling_factor;
    SHERPA_ONNX_LOGE("CANARY DecodeStream: num_feature_frames = %d", num_feature_frames);

    std::vector<int32_t> tokens = {max_token_id};
    std::vector<float> token_log_probs = {confidence};

    // Assume 30 tokens per second. It is to avoid the following for loop
    // running indefinitely.
    int32_t num_tokens =
        static_cast<int32_t>(num_feature_frames / 100.0 * 30) + 1;
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Max tokens to generate = %d", num_tokens);

    if (max_token_id == eos) {
      SHERPA_ONNX_LOGE("CANARY DecodeStream: WARNING - First token is EOS! Stopping immediately");
    }

    // Start from decoder_input.size() + 1 for proper position tracking
    int32_t tokens_generated = 0;
    for (int32_t i = decoder_input.size() + 1; i <= decoder_input.size() + num_tokens; ++i) {
      if (tokens.back() == eos) {
        SHERPA_ONNX_LOGE("CANARY DecodeStream: Hit EOS token at iteration %d, breaking", 
                         tokens_generated);
        break;
      }

      SHERPA_ONNX_LOGE("CANARY DecodeStream: Generating token %d (position %d)", 
                       tokens_generated + 1, i);
      
      std::tie(logits, decoder_states) =
          RunDecoder(tokens.back(), i, std::move(decoder_states),
                     View(&enc_states), View(&enc_mask));
      
      auto [next_token_id, next_confidence] = GetMaxTokenIdWithConfidence(&logits);
      SHERPA_ONNX_LOGE("CANARY DecodeStream: Generated token=%d, confidence=%.4f", 
                       next_token_id, next_confidence);
      
      tokens.push_back(next_token_id);
      token_log_probs.push_back(next_confidence);
      tokens_generated++;
    }

    SHERPA_ONNX_LOGE("CANARY DecodeStream: Generation complete. Generated %d tokens", 
                     tokens_generated);
    SHERPA_ONNX_LOGE("CANARY DecodeStream: tokens.size()=%zu, token_log_probs.size()=%zu", 
                     tokens.size(), token_log_probs.size());

    // remove the last eos token and its confidence
    if (!tokens.empty() && tokens.back() == eos) {
      SHERPA_ONNX_LOGE("CANARY DecodeStream: Removing final EOS token");
      tokens.pop_back();
      token_log_probs.pop_back();
    }

    SHERPA_ONNX_LOGE("CANARY DecodeStream: After EOS removal: tokens=%zu, probs=%zu", 
                     tokens.size(), token_log_probs.size());

    SHERPA_ONNX_LOGE("CANARY DecodeStream: Converting tokens to text");
    auto r = Convert(tokens, token_log_probs);
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Text length = %zu chars", r.text.length());
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Text = '%s'", r.text.c_str());

    SHERPA_ONNX_LOGE("CANARY DecodeStream: Applying inverse text normalization");
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    
    SHERPA_ONNX_LOGE("CANARY DecodeStream: Applying homophone replacer");
    r.text = ApplyHomophoneReplacer(std::move(r.text));

    SHERPA_ONNX_LOGE("CANARY DecodeStream: Setting result on stream");
    s->SetResult(r);
    SHERPA_ONNX_LOGE("CANARY DecodeStream: ========== END ==========");
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

  void SetConfig(const OfflineRecognizerConfig &config) override {
    config_.model_config.canary.src_lang = config.model_config.canary.src_lang;
    config_.model_config.canary.tgt_lang = config.model_config.canary.tgt_lang;
    config_.model_config.canary.use_pnc = config.model_config.canary.use_pnc;

    // we don't change the config_ in the base class
  }

 private:
  OfflineRecognitionResult Convert(const std::vector<int32_t> &tokens,
                                  const std::vector<float> &token_log_probs) const {
    SHERPA_ONNX_LOGE("CANARY Convert: Converting %zu tokens to text", tokens.size());
    
    OfflineRecognitionResult r;
    r.tokens.reserve(tokens.size());
    r.token_probs = token_log_probs;

    std::string text;
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
      int32_t token_id = tokens[idx];
      SHERPA_ONNX_LOGE("CANARY Convert: Token[%zu] = %d", idx, token_id);
      
      if (!symbol_table_.Contains(token_id)) {
        SHERPA_ONNX_LOGE("CANARY Convert: WARNING - Token %d not in symbol table!", token_id);
        continue;
      }

      const auto &s = symbol_table_[token_id];
      SHERPA_ONNX_LOGE("CANARY Convert: Token %d -> '%s'", token_id, s.c_str());
      text += s;
      r.tokens.push_back(s);
    }

    r.text = std::move(text);
    SHERPA_ONNX_LOGE("CANARY Convert: Final text = '%s'", r.text.c_str());
    SHERPA_ONNX_LOGE("CANARY Convert: Returning %zu tokens, %zu probs", 
                     r.tokens.size(), r.token_probs.size());

    return r;
  }

  std::pair<int32_t, float> GetMaxTokenIdWithConfidence(Ort::Value *logits) const {
    SHERPA_ONNX_LOGE("CANARY GetMaxToken: Starting");
    
    // logits is of shape (1, 1, vocab_size)
    auto meta = model_->GetModelMetadata();
    const float *p_logits = logits->GetTensorData<float>();
    
    SHERPA_ONNX_LOGE("CANARY GetMaxToken: vocab_size = %d", meta.vocab_size);

    // Find max for numerical stability
    float max_logit = *std::max_element(p_logits, p_logits + meta.vocab_size);
    SHERPA_ONNX_LOGE("CANARY GetMaxToken: max_logit = %.4f", max_logit);
    
    // Compute log_softmax
    float sum_exp = 0.0f;
    for (int32_t i = 0; i < meta.vocab_size; ++i) {
      sum_exp += std::exp(p_logits[i] - max_logit);
    }
    float log_sum = max_logit + std::log(sum_exp);
    SHERPA_ONNX_LOGE("CANARY GetMaxToken: log_sum = %.4f", log_sum);
    
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

    SHERPA_ONNX_LOGE("CANARY GetMaxToken: Selected token %d with log_prob %.4f", 
                     max_token_id, max_log_prob);
    return {max_token_id, max_log_prob};
  }

  std::vector<Ort::Value> RunEncoder(OfflineStream *s) const {
    SHERPA_ONNX_LOGE("CANARY RunEncoder: Starting");
    
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = config_.feat_config.feature_dim;
    SHERPA_ONNX_LOGE("CANARY RunEncoder: feat_dim = %d", feat_dim);
    
    std::vector<float> f = s->GetFrames();
    SHERPA_ONNX_LOGE("CANARY RunEncoder: Got %zu floats from stream", f.size());
    
    if (f.empty()) {
      SHERPA_ONNX_LOGE("CANARY RunEncoder: ERROR - No frames from stream!");
      return {};
    }

    int32_t num_frames = f.size() / feat_dim;
    SHERPA_ONNX_LOGE("CANARY RunEncoder: num_frames = %d", num_frames);

    std::array<int64_t, 3> shape = {1, num_frames, feat_dim};
    SHERPA_ONNX_LOGE("CANARY RunEncoder: Creating tensor with shape [1, %d, %d]", 
                     num_frames, feat_dim);

    Ort::Value x = Ort::Value::CreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    int64_t x_length_scalar = num_frames;
    std::array<int64_t, 1> x_length_shape = {1};
    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &x_length_scalar, 1,
                                 x_length_shape.data(), x_length_shape.size());
    
    SHERPA_ONNX_LOGE("CANARY RunEncoder: Calling ForwardEncoder");
    auto result = model_->ForwardEncoder(std::move(x), std::move(x_length));
    SHERPA_ONNX_LOGE("CANARY RunEncoder: ForwardEncoder returned %zu outputs", result.size());
    
    return result;
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> RunDecoder(
      int32_t token, int32_t pos, std::vector<Ort::Value> decoder_states,
      Ort::Value enc_states, Ort::Value enc_mask) const {
    SHERPA_ONNX_LOGE("CANARY RunDecoder: token=%d, pos=%d", token, pos);
    
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> shape = {1, 2};
    std::array<int32_t, 2> _decoder_input = {token, pos};

    Ort::Value decoder_input = Ort::Value::CreateTensor(
        memory_info, _decoder_input.data(), _decoder_input.size(), shape.data(),
        shape.size());

    SHERPA_ONNX_LOGE("CANARY RunDecoder: Calling ForwardDecoder");
    auto result = model_->ForwardDecoder(std::move(decoder_input),
                                  std::move(decoder_states),
                                  std::move(enc_states), std::move(enc_mask));
    SHERPA_ONNX_LOGE("CANARY RunDecoder: ForwardDecoder returned");
    
    return result;
  }

  // see
  // https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/canary/test_180m_flash.py#L242
  std::vector<int32_t> GetInitialDecoderInput() const {
    SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: Starting");
    
    auto canary_config = config_.model_config.canary;
    const auto &meta = model_->GetModelMetadata();

    std::vector<int32_t> decoder_input(9);
    
    decoder_input[0] = symbol_table_["<|startofcontext|>"];
    SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [0] <|startofcontext|> = %d", decoder_input[0]);
    
    decoder_input[1] = symbol_table_["<|startoftranscript|>"];
    SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [1] <|startoftranscript|> = %d", decoder_input[1]);
    
    decoder_input[2] = symbol_table_["<|emo:undefined|>"];
    SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [2] <|emo:undefined|> = %d", decoder_input[2]);

    if (canary_config.src_lang.empty() ||
        !meta.lang2id.count(canary_config.src_lang)) {
      decoder_input[3] = meta.lang2id.at("en");
      SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [3] src_lang defaulted to 'en' = %d", 
                       decoder_input[3]);
    } else {
      decoder_input[3] = meta.lang2id.at(canary_config.src_lang);
      SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [3] src_lang '%s' = %d", 
                       canary_config.src_lang.c_str(), decoder_input[3]);
    }

    if (canary_config.tgt_lang.empty() ||
        !meta.lang2id.count(canary_config.tgt_lang)) {
      decoder_input[4] = meta.lang2id.at("en");
      SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [4] tgt_lang defaulted to 'en' = %d", 
                       decoder_input[4]);
    } else {
      decoder_input[4] = meta.lang2id.at(canary_config.tgt_lang);
      SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [4] tgt_lang '%s' = %d", 
                       canary_config.tgt_lang.c_str(), decoder_input[4]);
    }

    if (canary_config.use_pnc) {
      decoder_input[5] = symbol_table_["<|pnc|>"];
      SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [5] <|pnc|> = %d", decoder_input[5]);
    } else {
      decoder_input[5] = symbol_table_["<|nopnc|>"];
      SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [5] <|nopnc|> = %d", decoder_input[5]);
    }

    decoder_input[6] = symbol_table_["<|noitn|>"];
    SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [6] <|noitn|> = %d", decoder_input[6]);
    
    decoder_input[7] = symbol_table_["<|notimestamp|>"];
    SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [7] <|notimestamp|> = %d", decoder_input[7]);
    
    decoder_input[8] = symbol_table_["<|nodiarize|>"];
    SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: [8] <|nodiarize|> = %d", decoder_input[8]);

    SHERPA_ONNX_LOGE("CANARY GetInitialDecoderInput: Returning %zu tokens", decoder_input.size());
    return decoder_input;
  }

 private:
  void PostInit() {
    SHERPA_ONNX_LOGE("CANARY PostInit: Starting");
    
    auto &meta = model_->GetModelMetadata();
    config_.feat_config.feature_dim = meta.feat_dim;
    SHERPA_ONNX_LOGE("CANARY PostInit: Set feature_dim = %d", meta.feat_dim);

    config_.feat_config.nemo_normalize_type = meta.normalize_type;
    SHERPA_ONNX_LOGE("CANARY PostInit: Set normalize_type = '%s'", meta.normalize_type.c_str());

    config_.feat_config.dither = 0;
    config_.feat_config.remove_dc_offset = false;
    config_.feat_config.low_freq = 0;
    config_.feat_config.window_type = "hann";
    config_.feat_config.is_librosa = true;

    meta.lang2id["en"] = symbol_table_["<|en|>"];
    meta.lang2id["es"] = symbol_table_["<|es|>"];
    meta.lang2id["de"] = symbol_table_["<|de|>"];
    meta.lang2id["fr"] = symbol_table_["<|fr|>"];
    
    SHERPA_ONNX_LOGE("CANARY PostInit: Language IDs: en=%d, es=%d, de=%d, fr=%d",
                     meta.lang2id["en"], meta.lang2id["es"], 
                     meta.lang2id["de"], meta.lang2id["fr"]);

    if (symbol_table_.NumSymbols() != meta.vocab_size) {
      SHERPA_ONNX_LOGE("CANARY PostInit: ERROR - symbol_table has %d symbols but vocab_size is %d",
                       symbol_table_.NumSymbols(), meta.vocab_size);
      SHERPA_ONNX_EXIT(-1);
    }
    
    SHERPA_ONNX_LOGE("CANARY PostInit: Complete. vocab_size=%d", meta.vocab_size);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineCanaryModel> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CANARY_IMPL_H_