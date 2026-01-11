// sherpa-onnx/csrc/axera/offline-whisper-model-axera.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/offline-whisper-model-axera.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
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

#include "ax_engine_api.h"  // NOLINT
#include "ax_sys_api.h"     // NOLINT
#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"
#include "sherpa-onnx/csrc/axera/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

// masked positions: 1, unmasked: 0
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

class OfflineWhisperModelAxera::Impl {
 public:
  ~Impl() {
    FreeIO(&enc_io_);
    FreeIO(&dec_io_);
    if (enc_handle_) AX_ENGINE_DestroyHandle(enc_handle_);
    if (dec_handle_) AX_ENGINE_DestroyHandle(dec_handle_);
  }

  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    auto enc_buf = ReadFile(config_.whisper.encoder);
    auto dec_buf = ReadFile(config_.whisper.decoder);
    Init(enc_buf.data(), enc_buf.size(), dec_buf.data(), dec_buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    auto enc_buf = ReadFile(mgr, config_.whisper.encoder);
    auto dec_buf = ReadFile(mgr, config_.whisper.decoder);
    Init(enc_buf.data(), enc_buf.size(), dec_buf.data(), dec_buf.size());
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
          "model accepting longer audios or use VAD to cut audio.");
      num_frames = num_frames_;
    }

    // at most 6 tokens per second
    int32_t num_possible_tokens = static_cast<int32_t>(num_frames / 100.0 * 6);
    num_possible_tokens =
        std::min<int32_t>(num_possible_tokens, n_text_ctx_ / 2);

    // pad to encoder expected length
    features.resize(num_frames_ * feat_dim_, 0);

    // (num_frames_, feat_dim_) -> (feat_dim_, num_frames_)
    features = Transpose(features.data(), num_frames_, feat_dim_);

    RunEncoder(std::move(features));  // fill cross_kv_cpu_

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
  void Init(void *enc_model_data, size_t enc_len, void *dec_model_data,
            size_t dec_len) {
    InitContext(enc_model_data, enc_len, config_.debug, &enc_handle_);
    InitInputOutputAttrs(enc_handle_, config_.debug, &enc_io_info_);
    PrepareIO(enc_io_info_, &enc_io_, config_.debug);

    InitContext(dec_model_data, dec_len, config_.debug, &dec_handle_);
    InitInputOutputAttrs(dec_handle_, config_.debug, &dec_io_info_);
    PrepareIO(dec_io_info_, &dec_io_, config_.debug);

    PostInitEncoder();
    PostInitDecoder();
    InitSotSequence();

    // allocate CPU buffers
    cross_kv_cpu_.resize(n_text_layer_ * 2);
    for (int32_t i = 0; i < n_text_layer_ * 2; ++i) {
      cross_kv_cpu_[i].resize(num_out_frames_ * n_text_state_);
    }
    self_kv_cpu_.resize(n_text_layer_ * 2 * n_text_ctx_ * n_text_state_);
    delta_kv_cpu_.resize(n_text_layer_ * 2 * n_text_state_);
    logits_cpu_.resize(vocab_size_);

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Axera Whisper init done.");
      SHERPA_ONNX_LOGE("  feat_dim_ = %d", feat_dim_);
      SHERPA_ONNX_LOGE("  num_frames_ = %d", num_frames_);
      SHERPA_ONNX_LOGE("  num_out_frames_ = %d", num_out_frames_);
      SHERPA_ONNX_LOGE("  n_text_layer_ = %d", n_text_layer_);
      SHERPA_ONNX_LOGE("  n_text_ctx_ = %d", n_text_ctx_);
      SHERPA_ONNX_LOGE("  n_text_state_ = %d", n_text_state_);
      SHERPA_ONNX_LOGE("  vocab_size_ = %d", vocab_size_);
    }
  }

  // encoder: input 0 mel (float32), outputs: cross_kv (2L tensors)
  void PostInitEncoder() {
    if (!enc_io_info_ || enc_io_info_->nInputSize < 1) {
      SHERPA_ONNX_LOGE("Encoder has no inputs!");
      SHERPA_ONNX_EXIT(-1);
    }
    if (enc_io_info_->nOutputSize < 2) {
      SHERPA_ONNX_LOGE("Encoder outputs are unexpected!");
      SHERPA_ONNX_EXIT(-1);
    }

    // Use input name to parse model type: "<tiny/base/...>-mel"
    model_type_ = ParseWhisperModelFromString(enc_io_info_->pInputs[0].pName);

    const auto &in0 = enc_io_info_->pInputs[0];
    // expect shape: [1, feat_dim, num_frames]
    if (in0.nShapeSize != 3 || in0.pShape[0] != 1) {
      SHERPA_ONNX_LOGE("Encoder input shape is unexpected.");
      SHERPA_ONNX_EXIT(-1);
    }
    feat_dim_ = in0.pShape[1];
    num_frames_ = in0.pShape[2];

    int32_t num_out = static_cast<int32_t>(enc_io_info_->nOutputSize);
    n_text_layer_ = num_out / 2;

    const auto &out0 = enc_io_info_->pOutputs[0];
    // expect [1, num_out_frames, n_text_state]
    if (out0.nShapeSize < 3 || out0.pShape[0] != 1) {
      SHERPA_ONNX_LOGE("Encoder output shape is unexpected.");
      SHERPA_ONNX_EXIT(-1);
    }
    num_out_frames_ = out0.pShape[1];
    n_text_state_ = out0.pShape[out0.nShapeSize - 1];
  }

  // decoder: inputs: token + self_kv(2L) + cross_kv(2L) + offset + mask
  // outputs: logits + delta_kv(2L)
  void PostInitDecoder() {
    if (!dec_io_info_) {
      SHERPA_ONNX_LOGE("Decoder io_info is null");
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t expected_num_inputs = 1 + 2 * n_text_layer_ + 2 * n_text_layer_ + 2;
    if (static_cast<int32_t>(dec_io_info_->nInputSize) != expected_num_inputs) {
      SHERPA_ONNX_LOGE("Decoder expects %d inputs. Actual: %u",
                       expected_num_inputs, dec_io_info_->nInputSize);
      SHERPA_ONNX_EXIT(-1);
    }

    // infer n_text_ctx from self_kv input #1 shape: [1, n_text_ctx,
    // n_text_state]
    const auto &self0 = dec_io_info_->pInputs[1];
    if (self0.nShapeSize != 3 || self0.pShape[0] != 1) {
      SHERPA_ONNX_LOGE("Decoder self_kv input shape is unexpected.");
      SHERPA_ONNX_EXIT(-1);
    }
    n_text_ctx_ = self0.pShape[1];
    if (self0.pShape[2] != n_text_state_) {
      SHERPA_ONNX_LOGE("Expect n_text_state_ %d. Given: %d", n_text_state_,
                       self0.pShape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    // mask buffer on CPU: [token, offset, mask...]
    token_offset_mask_cpu_.resize(1 + 1 + n_text_ctx_);

    // output[0] logits: [..., vocab_size]
    const auto &logits = dec_io_info_->pOutputs[0];
    if (logits.nShapeSize < 1) {
      SHERPA_ONNX_LOGE("Decoder logits output shape is unexpected.");
      SHERPA_ONNX_EXIT(-1);
    }
    vocab_size_ = logits.pShape[logits.nShapeSize - 1];

    // outputs should be 1 + 2L
    if (static_cast<int32_t>(dec_io_info_->nOutputSize) !=
        1 + 2 * n_text_layer_) {
      SHERPA_ONNX_LOGE("Decoder expects %d outputs. Actual: %u",
                       1 + 2 * n_text_layer_, dec_io_info_->nOutputSize);
      SHERPA_ONNX_EXIT(-1);
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
    // encoder input 0
    const auto &in0 = enc_io_info_->pInputs[0];
    if (in0.eDataType != AX_ENGINE_DT_FLOAT32) {
      SHERPA_ONNX_LOGE("Encoder input 0 expects float32");
      SHERPA_ONNX_EXIT(-1);
    }
    size_t bytes = in0.nSize;
    if (bytes != features.size() * sizeof(float)) {
      SHERPA_ONNX_LOGE("Encoder input bytes mismatch. expect %u, got %zu",
                       in0.nSize, features.size() * sizeof(float));
      SHERPA_ONNX_EXIT(-1);
    }
    std::memcpy(enc_io_.pInputs[0].pVirAddr, features.data(), bytes);

    auto ret = AX_ENGINE_RunSync(enc_handle_, &enc_io_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync(encoder) failed, ret=%d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    // fetch cross_kv outputs: 2L tensors
    int32_t num_out = static_cast<int32_t>(enc_io_info_->nOutputSize);
    if (num_out != n_text_layer_ * 2) {
      SHERPA_ONNX_LOGE("Unexpected encoder num outputs: %d", num_out);
      SHERPA_ONNX_EXIT(-1);
    }
    for (int32_t i = 0; i < num_out; ++i) {
      const auto &meta = enc_io_info_->pOutputs[i];
      auto &buf = enc_io_.pOutputs[i];
      size_t elems = meta.nSize / sizeof(float);
      if (static_cast<int32_t>(elems) != num_out_frames_ * n_text_state_) {
        SHERPA_ONNX_LOGE("Encoder output %d has unexpected size %zu", i, elems);
        SHERPA_ONNX_EXIT(-1);
      }
      cross_kv_cpu_[i].resize(elems);
      std::memcpy(cross_kv_cpu_[i].data(), buf.pVirAddr, meta.nSize);
    }
  }

  int32_t DetectLanguage() {
    RunDecoderOneStep(/*update_kv_cache*/ false);

    const auto &all_lang_ids = GetAllWhisperLanguageTokenIds();
    int32_t lang_id = all_lang_ids[0];
    float best = logits_cpu_[lang_id];
    for (int32_t i = 1; i < static_cast<int32_t>(all_lang_ids.size()); ++i) {
      int32_t id = all_lang_ids[i];
      float v = logits_cpu_[id];
      if (v > best) {
        best = v;
        lang_id = id;
      }
    }
    return lang_id;
  }

  int32_t RunDecoderOneStep(bool update_kv_cache) {
    int32_t token = token_offset_mask_cpu_[0];
    int32_t offset = token_offset_mask_cpu_[1];
    int32_t *mask = token_offset_mask_cpu_.data() + 2;

    // input 0: token (int32)
    {
      const auto &meta = dec_io_info_->pInputs[0];
      if (meta.nSize != sizeof(int32_t)) {
        SHERPA_ONNX_LOGE("Decoder token input bytes unexpected: %u",
                         meta.nSize);
        SHERPA_ONNX_EXIT(-1);
      }
      std::memcpy(dec_io_.pInputs[0].pVirAddr, &token, sizeof(int32_t));
    }

    // inputs 1..2L: self_kv (float32)
    for (int32_t i = 0; i < 2 * n_text_layer_; ++i) {
      const auto &meta = dec_io_info_->pInputs[1 + i];
      size_t need_bytes =
          static_cast<size_t>(n_text_ctx_) * n_text_state_ * sizeof(float);
      if (meta.nSize != need_bytes) {
        SHERPA_ONNX_LOGE(
            "Decoder self_kv[%d] bytes mismatch. expect %zu got %u", i,
            need_bytes, meta.nSize);
        SHERPA_ONNX_EXIT(-1);
      }
      const float *p = self_kv_cpu_.data() + i * (n_text_ctx_ * n_text_state_);
      std::memcpy(dec_io_.pInputs[1 + i].pVirAddr, p, need_bytes);
    }

    // cross_kv base
    int32_t cross_base = 1 + 2 * n_text_layer_;
    for (int32_t i = 0; i < 2 * n_text_layer_; ++i) {
      const auto &meta = dec_io_info_->pInputs[cross_base + i];
      size_t need_bytes =
          static_cast<size_t>(num_out_frames_) * n_text_state_ * sizeof(float);
      if (meta.nSize != need_bytes) {
        SHERPA_ONNX_LOGE(
            "Decoder cross_kv[%d] bytes mismatch. expect %zu got %u", i,
            need_bytes, meta.nSize);
        SHERPA_ONNX_EXIT(-1);
      }
      std::memcpy(dec_io_.pInputs[cross_base + i].pVirAddr,
                  cross_kv_cpu_[i].data(), need_bytes);
    }

    // offset (int32): index = 1 + 4L
    {
      int32_t idx = 1 + 4 * n_text_layer_;
      const auto &meta = dec_io_info_->pInputs[idx];
      if (meta.nSize != sizeof(int32_t)) {
        SHERPA_ONNX_LOGE("Decoder offset input bytes unexpected: %u",
                         meta.nSize);
        SHERPA_ONNX_EXIT(-1);
      }
      std::memcpy(dec_io_.pInputs[idx].pVirAddr, &offset, sizeof(int32_t));
    }

    // mask (int32[n_text_ctx]): index = 1 + 4L + 1
    {
      int32_t idx = 1 + 4 * n_text_layer_ + 1;
      const auto &meta = dec_io_info_->pInputs[idx];
      size_t need_bytes = static_cast<size_t>(n_text_ctx_) * sizeof(int32_t);
      if (meta.nSize != need_bytes) {
        SHERPA_ONNX_LOGE("Decoder mask bytes mismatch. expect %zu got %u",
                         need_bytes, meta.nSize);
        SHERPA_ONNX_EXIT(-1);
      }
      std::memcpy(dec_io_.pInputs[idx].pVirAddr, mask, need_bytes);
    }

    auto ret = AX_ENGINE_RunSync(dec_handle_, &dec_io_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync(decoder) failed, ret=%d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    // output 0: logits float[vocab_size]
    {
      const auto &meta = dec_io_info_->pOutputs[0];
      size_t elems = meta.nSize / sizeof(float);
      if (static_cast<int32_t>(elems) != vocab_size_) {
        SHERPA_ONNX_LOGE("Decoder logits size mismatch. expect %d got %zu",
                         vocab_size_, elems);
        SHERPA_ONNX_EXIT(-1);
      }
      logits_cpu_.resize(elems);
      std::memcpy(logits_cpu_.data(), dec_io_.pOutputs[0].pVirAddr, meta.nSize);
    }

    // output 1..2L: delta_kv float[n_text_state]
    if (update_kv_cache) {
      for (int32_t i = 0; i < 2 * n_text_layer_; ++i) {
        const auto &meta = dec_io_info_->pOutputs[1 + i];
        size_t need_bytes = static_cast<size_t>(n_text_state_) * sizeof(float);
        if (meta.nSize != need_bytes) {
          SHERPA_ONNX_LOGE(
              "Decoder delta_kv[%d] bytes mismatch. expect %zu got %u", i,
              need_bytes, meta.nSize);
          SHERPA_ONNX_EXIT(-1);
        }
        float *dst = delta_kv_cpu_.data() + i * n_text_state_;
        std::memcpy(dst, dec_io_.pOutputs[1 + i].pVirAddr, need_bytes);
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

 private:
  std::mutex mutex_;
  AxEngineGuard ax_engine_guard_;
  OfflineModelConfig config_;

  // encoder
  AX_ENGINE_HANDLE enc_handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *enc_io_info_ = nullptr;
  AX_ENGINE_IO_T enc_io_;

  // decoder
  AX_ENGINE_HANDLE dec_handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *dec_io_info_ = nullptr;
  AX_ENGINE_IO_T dec_io_;

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

OfflineWhisperModelAxera::~OfflineWhisperModelAxera() = default;

OfflineWhisperModelAxera::OfflineWhisperModelAxera(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineWhisperModelAxera::OfflineWhisperModelAxera(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineWhisperDecoderResult OfflineWhisperModelAxera::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineWhisperModelAxera::FeatureDim() const {
  return impl_->FeatureDim();
}

#if __ANDROID_API__ >= 9
template OfflineWhisperModelAxera::OfflineWhisperModelAxera(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineWhisperModelAxera::OfflineWhisperModelAxera(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx