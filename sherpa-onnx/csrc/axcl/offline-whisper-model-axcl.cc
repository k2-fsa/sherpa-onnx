// sherpa-onnx/csrc/axcl/offline-whisper-model-axcl.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include "sherpa-onnx/csrc/axcl/offline-whisper-model-axcl.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/axcl/axcl-model.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

// masked positions: 1
// unmasked positions: 0
static void UpdateCausalMask(int32_t offset, int32_t capacity, int32_t *p) {
  std::fill(p, p + offset, 0);
  std::fill(p + offset, p + capacity, 1);
}

static WhisperModelType ParseWhisperModelFromString(const std::string &s) {
  auto pos = s.find('-');
  if (pos == std::string::npos) {
    SHERPA_ONNX_LOGE("Unexpected model input '%s'", s.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  if (s.substr(pos + 1) != "mel") {
    SHERPA_ONNX_LOGE("Unexpected model input '%s'", s.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  if (pos == 0) {
    SHERPA_ONNX_LOGE("Empty model name in '%s'", s.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  return ParseWhisperModelType(s.substr(0, pos));
}

class OfflineWhisperModelAxcl::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    InitEncoder(config_.whisper.encoder);
    InitDecoder(config_.whisper.decoder);
    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(mgr, config_.whisper.encoder);
      InitEncoder(buf.data(), buf.size());
    }
    {
      auto buf = ReadFile(mgr, config_.whisper.decoder);
      InitDecoder(buf.data(), buf.size());
    }
    PostInit();
  }

  OfflineWhisperDecoderResult Run(std::vector<float> features) {
    std::lock_guard<std::mutex> lock(mutex_);

    OfflineWhisperDecoderResult r;
    if (features.empty()) return r;

    int32_t num_frames = static_cast<int32_t>(features.size() / feat_dim_);
    if (num_frames > num_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          num_frames, num_frames_);
      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios or use VAD to cut your audio into "
          "small chunks.");
      num_frames = num_frames_;
    }

    // assume at most 6 tokens per second
    int32_t num_possible_tokens = static_cast<int32_t>(num_frames / 100.0 * 6);
    num_possible_tokens =
        std::min<int32_t>(num_possible_tokens, n_text_ctx_ / 2);

    // pad to encoder expected length
    features.resize(num_frames_ * feat_dim_, 0);

    // (num_frames_, feat_dim_) -> (feat_dim_, num_frames_)
    features = Transpose(features.data(), num_frames_, feat_dim_);

    RunEncoder(std::move(features));  // fills cross_kv_cpu_

    // init self kv cache (CPU)
    std::fill(self_kv_cpu_.begin(), self_kv_cpu_.end(), 0.0f);

    std::vector<int32_t> sot_sequence(sot_sequence_);
    if (IsMultilingual(model_type_)) {
      if (config_.whisper.task == "translate") {
        sot_sequence[2] = translate_;
      } else if (config_.whisper.task != "transcribe") {
        SHERPA_ONNX_LOGE(
            "Valid task values are: translate, transcribe. Given: '%s'",
            config_.whisper.task.c_str());
        SHERPA_ONNX_EXIT(-1);
      }

      if (!config_.whisper.language.empty()) {
        int32_t lang_id = GetWhisperLanguageTokenId(config_.whisper.language);
        if (lang_id < 0) {
          SHERPA_ONNX_LOGE("Unsupported language: '%s'",
                           config_.whisper.language.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
        r.lang = config_.whisper.language;
        sot_sequence[1] = lang_id;
      } else {
        // detect language
        if (config_.debug) SHERPA_ONNX_LOGE("Detecting language.");

        token_offset_mask_cpu_[0] = sot_sequence_[0];
        token_offset_mask_cpu_[1] = 0;
        UpdateCausalMask(0, n_text_ctx_, token_offset_mask_cpu_.data() + 2);

        int32_t lang_id = DetectLanguage();
        r.lang = GetWhisperLanguageCode(lang_id);

        if (config_.debug)
          SHERPA_ONNX_LOGE("Detected Language: %s", r.lang.c_str());
        sot_sequence[1] = lang_id;
      }
    }

    int32_t &token = token_offset_mask_cpu_[0];
    int32_t &offset = token_offset_mask_cpu_[1];
    offset = 0;

    int32_t *p_mask = token_offset_mask_cpu_.data() + 2;
    UpdateCausalMask(offset, n_text_ctx_, p_mask);

    for (int32_t i = 0; i < static_cast<int32_t>(sot_sequence.size()); ++i) {
      token = sot_sequence[i];
      token = RunDecoderOneStep(/*update_kv_cache*/ true);
      p_mask[offset] = 0;
      offset += 1;
    }

    if (token == eot_) return r;

    r.tokens.reserve(num_possible_tokens);
    while (offset < num_possible_tokens && token != eot_) {
      r.tokens.push_back(token);
      token = RunDecoderOneStep(/*update_kv_cache*/ true);
      p_mask[offset] = 0;
      offset += 1;
    }

    return r;
  }

  int32_t FeatureDim() const { return feat_dim_; }

 private:
  void PostInit() {
    if (!encoder_model_->IsInitialized()) {
      SHERPA_ONNX_LOGE("Failed to initialize encoder: '%s'",
                       config_.whisper.encoder.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    if (!decoder_model_->IsInitialized()) {
      SHERPA_ONNX_LOGE("Failed to initialize decoder: '%s'",
                       config_.whisper.decoder.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    PostInitEncoder();
    PostInitDecoder();
    InitSotSequence();

    // allocate CPU buffers
    // cross_kv: n_text_layer_*2 * (num_out_frames_ * n_text_state_)
    cross_kv_cpu_.resize(n_text_layer_ * 2);
    for (int32_t i = 0; i < n_text_layer_ * 2; ++i) {
      cross_kv_cpu_[i].resize(num_out_frames_ * n_text_state_);
    }

    // self_kv cache: n_text_layer_*2 * (n_text_ctx_ * n_text_state_)
    self_kv_cpu_.resize(n_text_layer_ * 2 * n_text_ctx_ * n_text_state_);

    // delta_kv: n_text_layer_*2 * (n_text_state_)
    delta_kv_cpu_.resize(n_text_layer_ * 2 * n_text_state_);

    logits_cpu_.resize(vocab_size_);
  }

  void PostInitEncoder() {
    const auto &names = encoder_model_->InputTensorNames();
    if (names.empty()) {
      SHERPA_ONNX_LOGE("Encoder has no inputs!");
      SHERPA_ONNX_EXIT(-1);
    }
    model_type_ = ParseWhisperModelFromString(names[0]);

    // input mel shape: (1, feat_dim, num_frames)
    auto mel_shape = encoder_model_->TensorShape(encoder_model_->InputName(0));
    if (mel_shape.size() != 3 || mel_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Encoder input shape is unexpected.");
      SHERPA_ONNX_EXIT(-1);
    }
    feat_dim_ = mel_shape[1];
    num_frames_ = mel_shape[2];

    // outputs: cross_kv, count = n_text_layer_*2
    int32_t num_out = encoder_model_->NumOutputs();
    n_text_layer_ = num_out / 2;

    auto y_shape = encoder_model_->TensorShape(encoder_model_->OutputName(0));
    // expect (1, num_out_frames, n_text_state)
    if (y_shape.size() < 3) {
      SHERPA_ONNX_LOGE("Encoder output shape is unexpected.");
      SHERPA_ONNX_EXIT(-1);
    }
    num_out_frames_ = y_shape[1];
    n_text_state_ = y_shape.back();

    if (config_.debug) {
      SHERPA_ONNX_LOGE("model type: %s", ToString(model_type_).c_str());
      SHERPA_ONNX_LOGE("feat_dim_: %d", feat_dim_);
      SHERPA_ONNX_LOGE("num_frames_: %d", num_frames_);
      SHERPA_ONNX_LOGE("num_out_frames_: %d", num_out_frames_);
      SHERPA_ONNX_LOGE("n_text_layer_: %d", n_text_layer_);
      SHERPA_ONNX_LOGE("n_text_state_: %d", n_text_state_);
    }
  }

  void PostInitDecoder() {
    // inputs: token + self_kv(2L) + cross_kv(2L) + offset + mask
    int32_t expected_num_inputs = 1 + 2 * n_text_layer_ + 2 * n_text_layer_ + 2;
    if (decoder_model_->NumInputs() != expected_num_inputs) {
      SHERPA_ONNX_LOGE("Decoder expects %d inputs. Actual: %d",
                       expected_num_inputs, decoder_model_->NumInputs());
      SHERPA_ONNX_EXIT(-1);
    }

    // infer n_text_ctx from self_kv input #1 shape: (1, n_text_ctx,
    // n_text_state)
    auto s = decoder_model_->TensorShape(decoder_model_->InputName(1));
    if (s.size() != 3 || s[0] != 1) {
      SHERPA_ONNX_LOGE("Decoder self_kv input shape is unexpected.");
      SHERPA_ONNX_EXIT(-1);
    }
    n_text_ctx_ = s[1];
    if (s[2] != n_text_state_) {
      SHERPA_ONNX_LOGE("Expect n_text_state_ %d. Given: %d", n_text_state_,
                       s[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    token_offset_mask_cpu_.resize(1 + 1 + n_text_ctx_);

    // output[0] logits: (..., vocab_size)
    auto out0 = decoder_model_->TensorShape(decoder_model_->OutputName(0));
    vocab_size_ = out0.back();

    if (config_.debug) {
      SHERPA_ONNX_LOGE("n_text_ctx_: %d", n_text_ctx_);
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
    }
  }

  void InitSotSequence() {
    switch (model_type_) {
      case WhisperModelType::TinyEn:
      case WhisperModelType::BaseEn:
      case WhisperModelType::SmallEn:
      case WhisperModelType::MediumEn:
        sot_sequence_ = {50257, 50362};
        eot_ = 50256;
        break;
      case WhisperModelType::Tiny:
      case WhisperModelType::Base:
      case WhisperModelType::Small:
      case WhisperModelType::Medium:
      case WhisperModelType::Large:
        sot_sequence_ = {50258, 50259, 50359, 50363};
        eot_ = 50257;
        translate_ = 50358;
        break;
      default:
        SHERPA_ONNX_LOGE("Unsupported model type: '%s'",
                         ToString(model_type_).c_str());
        SHERPA_ONNX_EXIT(-1);
    }
  }

  void RunEncoder(std::vector<float> features) {
    // encoder input 0 is mel
    const std::string &mel_name = encoder_model_->InputName(0);
    if (!encoder_model_->SetInputTensorData(
            mel_name, features.data(), static_cast<int32_t>(features.size()))) {
      SHERPA_ONNX_LOGE("Failed to set encoder input '%s'", mel_name.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!encoder_model_->Run()) {
      SHERPA_ONNX_LOGE("Failed to run encoder");
      SHERPA_ONNX_EXIT(-1);
    }

    // fetch cross_kv outputs in order
    int32_t num_out = encoder_model_->NumOutputs();
    if (num_out != n_text_layer_ * 2) {
      SHERPA_ONNX_LOGE("Unexpected encoder num outputs: %d", num_out);
      SHERPA_ONNX_EXIT(-1);
    }

    for (int32_t i = 0; i < num_out; ++i) {
      const std::string &out_name = encoder_model_->OutputName(i);
      cross_kv_cpu_[i] = encoder_model_->GetOutputTensorData(out_name);
      // expect size = num_out_frames_ * n_text_state_
      if (static_cast<int32_t>(cross_kv_cpu_[i].size()) !=
          num_out_frames_ * n_text_state_) {
        SHERPA_ONNX_LOGE("Encoder output '%s' has unexpected size %d",
                         out_name.c_str(),
                         static_cast<int32_t>(cross_kv_cpu_[i].size()));
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  int32_t DetectLanguage() {
    RunDecoderOneStep(/*update_kv_cache*/ false);

    const auto &all_lang_ids = GetAllWhisperLanguageTokenIds();
    int32_t lang_id = all_lang_ids[0];
    float this_logit = logits_cpu_[lang_id];
    for (int32_t i = 1; i != static_cast<int32_t>(all_lang_ids.size()); ++i) {
      int32_t id = all_lang_ids[i];
      float p = logits_cpu_[id];
      if (p > this_logit) {
        this_logit = p;
        lang_id = id;
      }
    }
    return lang_id;
  }

  int32_t RunDecoderOneStep(bool update_kv_cache) {
    // decoder inputs are in this order (same as ACL code by index):
    // 0 token
    // 1..2L self_kv
    // (1+2L)..(1+4L-1) cross_kv
    // last-2 offset
    // last-1 mask

    int32_t token = token_offset_mask_cpu_[0];
    int32_t offset = token_offset_mask_cpu_[1];
    int32_t *mask = token_offset_mask_cpu_.data() + 2;

    // set token
    {
      const std::string &name = decoder_model_->InputName(0);
      if (!decoder_model_->SetInputTensorData(name, &token, 1)) {
        SHERPA_ONNX_LOGE("Failed to set decoder input '%s'", name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    // set self_kv (2L tensors)
    for (int32_t i = 0; i < 2 * n_text_layer_; ++i) {
      const std::string &name = decoder_model_->InputName(1 + i);
      const float *p = self_kv_cpu_.data() + i * (n_text_ctx_ * n_text_state_);
      int32_t n = n_text_ctx_ * n_text_state_;
      if (!decoder_model_->SetInputTensorData(name, p, n)) {
        SHERPA_ONNX_LOGE("Failed to set decoder input '%s'", name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    // set cross_kv (2L tensors)
    int32_t cross_base = 1 + 2 * n_text_layer_;
    for (int32_t i = 0; i < 2 * n_text_layer_; ++i) {
      const std::string &name = decoder_model_->InputName(cross_base + i);
      const float *p = cross_kv_cpu_[i].data();
      int32_t n = num_out_frames_ * n_text_state_;
      if (!decoder_model_->SetInputTensorData(name, p, n)) {
        SHERPA_ONNX_LOGE("Failed to set decoder input '%s'", name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    // set offset
    {
      const std::string &name =
          decoder_model_->InputName(1 + 4 * n_text_layer_);
      if (!decoder_model_->SetInputTensorData(name, &offset, 1)) {
        SHERPA_ONNX_LOGE("Failed to set decoder input '%s'", name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    // set mask
    {
      const std::string &name =
          decoder_model_->InputName(1 + 4 * n_text_layer_ + 1);
      if (!decoder_model_->SetInputTensorData(name, mask, n_text_ctx_)) {
        SHERPA_ONNX_LOGE("Failed to set decoder input '%s'", name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    if (!decoder_model_->Run()) {
      SHERPA_ONNX_LOGE("Failed to run decoder");
      SHERPA_ONNX_EXIT(-1);
    }

    // get logits (output 0)
    {
      const std::string &name = decoder_model_->OutputName(0);
      logits_cpu_ = decoder_model_->GetOutputTensorData(name);
      if (static_cast<int32_t>(logits_cpu_.size()) != vocab_size_) {
        SHERPA_ONNX_LOGE("Decoder logits has unexpected size %d",
                         static_cast<int32_t>(logits_cpu_.size()));
        SHERPA_ONNX_EXIT(-1);
      }
    }

    // get delta_kv outputs: 2L tensors, each size n_text_state_
    if (update_kv_cache) {
      for (int32_t i = 0; i < 2 * n_text_layer_; ++i) {
        const std::string &name = decoder_model_->OutputName(1 + i);
        auto v = decoder_model_->GetOutputTensorData(name);
        if (static_cast<int32_t>(v.size()) != n_text_state_) {
          SHERPA_ONNX_LOGE("Decoder delta_kv '%s' has unexpected size %d",
                           name.c_str(), static_cast<int32_t>(v.size()));
          SHERPA_ONNX_EXIT(-1);
        }
        std::copy(v.begin(), v.end(),
                  delta_kv_cpu_.begin() + i * n_text_state_);
      }

      // update self kv cache at position offset
      for (int32_t i = 0; i < 2 * n_text_layer_; ++i) {
        float *dst = self_kv_cpu_.data() + i * (n_text_ctx_ * n_text_state_) +
                     offset * n_text_state_;
        const float *src = delta_kv_cpu_.data() + i * n_text_state_;
        std::copy(src, src + n_text_state_, dst);
      }
    }

    return MaxElementIndex(logits_cpu_.data(),
                           static_cast<int32_t>(logits_cpu_.size()));
  }

  void InitEncoder(const std::string &filename) {
    encoder_model_ = std::make_unique<AxclModel>(filename);
  }
  void InitEncoder(void *data, size_t size) {
    encoder_model_ = std::make_unique<AxclModel>(data, size);
  }
  void InitDecoder(const std::string &filename) {
    decoder_model_ = std::make_unique<AxclModel>(filename);
  }
  void InitDecoder(void *data, size_t size) {
    decoder_model_ = std::make_unique<AxclModel>(data, size);
  }

 private:
  std::mutex mutex_;
  OfflineModelConfig config_;

  std::unique_ptr<AxclModel> encoder_model_;
  std::unique_ptr<AxclModel> decoder_model_;

  WhisperModelType model_type_;

  int32_t feat_dim_ = 0;
  int32_t num_frames_ = 0;
  int32_t num_out_frames_ = 0;
  int32_t n_text_layer_ = 0;
  int32_t n_text_ctx_ = 0;
  int32_t n_text_state_ = 0;
  int32_t vocab_size_ = 0;

  // CPU buffers
  std::vector<std::vector<float>>
      cross_kv_cpu_;                 // [2L][num_out_frames*n_text_state]
  std::vector<float> self_kv_cpu_;   // [2L*n_text_ctx*n_text_state]
  std::vector<float> delta_kv_cpu_;  // [2L*n_text_state]
  std::vector<int32_t> token_offset_mask_cpu_;  // [token, offset, mask...]
  std::vector<float> logits_cpu_;

  std::vector<int32_t> sot_sequence_;
  int32_t eot_ = 0;
  int32_t translate_ = 0;
};

OfflineWhisperModelAxcl::OfflineWhisperModelAxcl(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineWhisperModelAxcl::OfflineWhisperModelAxcl(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineWhisperModelAxcl::~OfflineWhisperModelAxcl() = default;

OfflineWhisperDecoderResult OfflineWhisperModelAxcl::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineWhisperModelAxcl::FeatureDim() const {
  return impl_->FeatureDim();
}

#if __ANDROID_API__ >= 9
template OfflineWhisperModelAxcl::OfflineWhisperModelAxcl(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineWhisperModelAxcl::OfflineWhisperModelAxcl(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx