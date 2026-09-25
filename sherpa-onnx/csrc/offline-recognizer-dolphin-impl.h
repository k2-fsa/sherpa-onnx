// sherpa-onnx/csrc/offline-recognizer-dolphin-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_DOLPHIN_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_DOLPHIN_IMPL_H_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-dolphin-attention-model.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// Decoder header layout: <sos> <lang> <region> <asr> <timestamp> text... <eos>
static constexpr int32_t kDolphinHeaderLen = 5;

// Resolve before decoding. Invalid updates must not replace a valid prompt.
inline bool ResolveDolphinPrompt(const OfflineDolphinModelConfig &config,
                                 const SymbolTable &tokens, int32_t vocab_size,
                                 std::vector<int64_t> *prompt) {
  if (config.language.empty() && !config.region.empty()) {
    SHERPA_ONNX_LOGE("Dolphin region requires a language");
    return false;
  }

  std::vector<int64_t> resolved;
  for (int32_t field = 0; field != 2; ++field) {
    std::string code = field == 0 ? config.language : config.region;
    if (code.empty()) {
      continue;
    }
    // normalize case so "ZH"/"Cn" behave like "zh"/"CN"
    for (char &c : code) {
      if (field == 0 && c >= 'A' && c <= 'Z') {
        c += 'a' - 'A';
      } else if (field == 1 && c >= 'a' && c <= 'z') {
        c += 'A' - 'a';
      }
    }
    const bool valid_code =
        code.size() >= 2 && (field == 1 || code.size() <= 3) &&
        std::all_of(code.begin(), code.end(), [field](char c) {
          return field == 0 ? (c >= 'a' && c <= 'z') : (c >= 'A' && c <= 'Z');
        });
    auto it = tokens.sym2id().find("<" + code + ">");
    if (!valid_code || it == tokens.sym2id().end() || it->second < 0 ||
        it->second >= vocab_size || code == "sos" || code == "eos" ||
        code == "notimestamp" || code == "asr" || code == "blk" ||
        code == "unk") {
      SHERPA_ONNX_LOGE("Unsupported Dolphin language/region: %s", code.c_str());
      return false;
    }
    resolved.push_back(it->second);
  }
  *prompt = std::move(resolved);
  return true;
}

class OfflineRecognizerDolphinImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerDolphinImpl(const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineDolphinAttentionModel>(
            config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerDolphinImpl(Manager *mgr,
                               const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineDolphinAttentionModel>(
            mgr, config.model_config)) {
    Init();
  }

  void Init() {
    if (config_.decoding_method != "greedy_search") {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported at present for Dolphin attention "
          "decoder. Given %s",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    // feature settings are the same as the Dolphin CTC branch; see
    // OfflineRecognizerCtcImpl::Init
    config_.feat_config.low_freq = 0;
    config_.feat_config.high_freq = 8000;
    config_.feat_config.remove_dc_offset = false;
    config_.feat_config.dither = 0;
    config_.feat_config.preemph_coeff = 0;
    config_.feat_config.window_type = "hann";
    config_.feat_config.feature_dim = 80;
    config_.feat_config.is_librosa = true;
    config_.feat_config.frame_length_ms = 31.25;  // 16000/512 = 31.25
    config_.feat_config.snip_edges = false;

    // a tokens file that covers less than the decoder vocabulary would
    // silently emit garbage ids during decoding
    if (symbol_table_.NumSymbols() < model_->VocabSize()) {
      SHERPA_ONNX_LOGE(
          "tokens.txt has %d symbols, but the decoder expects vocab_size=%d. "
          "The tokens file likely does not match this decoder model.",
          symbol_table_.NumSymbols(), model_->VocabSize());
      SHERPA_ONNX_EXIT(-1);
    }

    sos_ = LookupTokenId("<sos>");
    eos_ = LookupTokenId("<eos>");
    notimestamp_ = LookupTokenId("<notimestamp>");
    if (!ResolveDolphinPrompt(config_.model_config.dolphin, symbol_table_,
                              model_->VocabSize(), &prompt_)) {
      SHERPA_ONNX_LOGE("Using automatic Dolphin language/region detection");
      config_.model_config.dolphin.language.clear();
      config_.model_config.dolphin.region.clear();
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  void SetConfig(const OfflineRecognizerConfig &config) override {
    const auto &dolphin = config.model_config.dolphin;
    if (!ResolveDolphinPrompt(dolphin, symbol_table_, model_->VocabSize(),
                              &prompt_)) {
      SHERPA_ONNX_LOGE("Ignoring invalid Dolphin configuration update");
      return;
    }
    // Sessions are not reloaded by SetConfig. Only the prompt is mutable.
    config_.model_config.dolphin.language = dolphin.language;
    config_.model_config.dolphin.region = dolphin.region;
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  int32_t LookupTokenId(const std::string &sym) const {
    const auto &sym2id = symbol_table_.sym2id();
    auto it = sym2id.find(sym);
    if (it == sym2id.end()) {
      SHERPA_ONNX_LOGE("tokens.txt does not contain the symbol '%s'",
                       sym.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    return it->second;
  }

  int32_t ArgMaxDecoderStep(Ort::Value &encoder_out,  // NOLINT
                            const std::vector<int64_t> &ys) const {
    std::array<int64_t, 2> shape{1, static_cast<int64_t>(ys.size())};
    Ort::Value tokens = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), shape.data(), shape.size());
    std::copy(ys.begin(), ys.end(), tokens.GetTensorMutableData<int64_t>());

    Ort::Value logp =
        model_->ForwardDecoderStep(encoder_out, std::move(tokens));
    const auto &logp_info = logp.GetTensorTypeAndShapeInfo();
    if (logp_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      SHERPA_ONNX_LOGE("Dolphin decoder must emit float32 logits");
      SHERPA_ONNX_EXIT(-1);
    }
    const float *p = logp.GetTensorData<float>();
    int64_t n = logp_info.GetElementCount();
    return static_cast<int32_t>(std::distance(p, std::max_element(p, p + n)));
  }

  void DecodeStream(OfflineStream *s) const {
    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();
    int32_t num_frames = f.size() / feat_dim;

    model_->NormalizeFeatures(f.data(), num_frames, feat_dim);

    std::array<int64_t, 3> shape{1, num_frames, feat_dim};
    Ort::Value feats = Ort::Value::CreateTensor<float>(
        model_->Allocator(), shape.data(), shape.size());
    std::copy(f.begin(), f.end(), feats.GetTensorMutableData<float>());

    std::array<int64_t, 1> len_shape{1};
    Ort::Value feats_len = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), len_shape.data(), len_shape.size());
    *feats_len.GetTensorMutableData<int64_t>() = num_frames;

    Ort::Value encoder_out =
        model_->ForwardEncoder(std::move(feats), std::move(feats_len));

    const auto &enc_shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
    if (enc_shape.size() < 2) {
      SHERPA_ONNX_LOGE("Dolphin encoder output rank %d is < 2",
                       static_cast<int32_t>(enc_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }
    int64_t enc_len = enc_shape[1];

    std::vector<int64_t> ys;
    ys.reserve(kDolphinHeaderLen + 16);
    ys.push_back(sos_);

    ys.insert(ys.end(), prompt_.begin(), prompt_.end());

    // header positions 1..4: language, region, task, timestamp
    while (static_cast<int64_t>(ys.size()) < kDolphinHeaderLen) {
      if (static_cast<int64_t>(ys.size()) == kDolphinHeaderLen - 1) {
        // timestamp token is forced to <notimestamp> since we don't
        // produce token-level timestamps on this path yet
        ys.push_back(notimestamp_);
      } else {
        ys.push_back(ArgMaxDecoderStep(encoder_out, ys));
      }
    }

    // The reference bounds the entire prefix, not only the text tokens.
    while (static_cast<int64_t>(ys.size()) <= enc_len) {
      int32_t next = ArgMaxDecoderStep(encoder_out, ys);
      ys.push_back(next);
      if (next == eos_) {
        break;
      }
    }

    OfflineRecognitionResult r;
    if (symbol_table_.Contains(ys[1])) {
      r.lang = StripAngleBrackets(symbol_table_[ys[1]]);
    }

    std::string text;
    for (int32_t i = kDolphinHeaderLen; i < static_cast<int32_t>(ys.size());
         ++i) {
      int64_t t = ys[i];
      if (t == eos_ || t == sos_ || !symbol_table_.Contains(t)) {
        continue;
      }
      std::string s = symbol_table_[t];
      text += s;
      r.tokens.push_back(s);
    }

    text = ApplyInverseTextNormalization(text);
    text = ApplyHomophoneReplacer(std::move(text));
    r.text = std::move(text);
    s->SetResult(r);
  }

  static std::string StripAngleBrackets(const std::string &sym) {
    if (sym.size() > 2 && sym.front() == '<' && sym.back() == '>') {
      return sym.substr(1, sym.size() - 2);
    }
    return sym;
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineDolphinAttentionModel> model_;
  std::vector<int64_t> prompt_;

  int32_t sos_ = -1;
  int32_t eos_ = -1;
  int32_t notimestamp_ = -1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_DOLPHIN_IMPL_H_
