// sherpa-onnx/csrc/ascend/offline-whisper-model-ascend.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include "sherpa-onnx/csrc/ascend/offline-whisper-model-ascend.h"

#include <algorithm>
#include <array>
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

#include "sherpa-onnx/csrc/ascend/macros.h"
#include "sherpa-onnx/csrc/ascend/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

// masked positions: 1
// unmasked positions: 0
static std::vector<int32_t> CreateCausalMask(int32_t n, int32_t capacity) {
  std::vector<int32_t> mask(capacity, 1);
  std::fill(mask.data(), mask.data() + n, 0);

  return mask;
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

class OfflineWhisperModelAscend::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    PreInit();

    InitEncoder(config_.whisper.encoder);
    InitDecoder(config_.whisper.decoder);

    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    SHERPA_ONNX_LOGE("Not implemented");
    SHERPA_ONNX_EXIT(-1);
  }

  OfflineWhisperDecoderResult Run(std::vector<float> features) {
    // TODO(fangjun): Support multi clients
    std::lock_guard<std::mutex> lock(mutex_);
    if (features.empty()) {
      return {};
    }

    int32_t num_frames = features.size() / feat_dim_;
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
    int32_t num_possible_tokens = num_frames / 100.0 * 6;
    num_possible_tokens =
        std::min<int32_t>(num_possible_tokens, n_text_ctx_ / 2);

    features.resize(num_frames_ * feat_dim_);

    // (num_frames_, feat_dim_) -> (feat_dim_, num_frames_)
    features = Transpose(features.data(), num_frames_, feat_dim_);

    SHERPA_ONNX_LOGE("features size: %d. %dx%d", (int)features.size(),
                     (int)features.size() / feat_dim_, feat_dim_);
    RunEncoder(std::move(features));

    // Note(fangjun): No need to intialize the self kv cache to 0

    int32_t &token = token_offset_mask_cpu_[0];
    int32_t &offset = token_offset_mask_cpu_[1];
    offset = 0;

    auto mask = CreateCausalMask(offset, n_text_ctx_);
    int32_t *p_mask = token_offset_mask_cpu_.data() + 2;
    std::copy(mask.begin(), mask.end(), p_mask);

    for (int32_t i = 0; i < sot_sequence_.size(); ++i) {
      token = sot_sequence_[i];
      token = RunDecoder();
      p_mask[offset] = 0;

      offset += 1;
      SHERPA_ONNX_LOGE("token: %d", token);
    }

    if (token == eot_) {
      return {};
    }
    OfflineWhisperDecoderResult r;
    r.tokens.reserve(num_possible_tokens);

    while (offset < num_possible_tokens && token != eot_) {
      r.tokens.push_back(token);
      token = RunDecoder();

      p_mask[offset] = 0;
      offset += 1;
    }

    return r;
  }

  int32_t FeatureDim() const { return feat_dim_; }

 private:
  void RunEncoder(std::vector<float> features) {
    aclError ret = aclrtMemcpy(features_ptr_, features.size() * sizeof(float),
                               features.data(), features.size() * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE);

    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    AclMdlDataset input_dataset;
    AclDataBuffer features_buf(features_ptr_, features.size() * sizeof(float));
    input_dataset.AddBuffer(features_buf);

    AclMdlDataset output_dataset;

    std::vector<AclDataBuffer> cross_kv_buffer;
    cross_kv_buffer.reserve(cross_kv_ptr_.size());
    for (auto p : cross_kv_ptr_) {
      AclDataBuffer tmp_buffer(p,
                               num_out_frames_ * n_text_state_ * sizeof(float));
      cross_kv_buffer.push_back(std::move(tmp_buffer));

      output_dataset.AddBuffer(cross_kv_buffer.back());
    }

    ret = aclmdlExecute(*encoder_model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlExecute");
  }

  int32_t RunDecoder() {
    aclError ret =
        aclrtMemcpy(token_ptr_, token_offset_mask_cpu_.size() * sizeof(int32_t),
                    token_offset_mask_cpu_.data(),
                    token_offset_mask_cpu_.size() * sizeof(int32_t),
                    ACL_MEMCPY_HOST_TO_DEVICE);

    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    AclMdlDataset input_dataset;
    AclDataBuffer token_buf(token_ptr_, sizeof(int32_t));
    input_dataset.AddBuffer(token_buf);

    std::vector<AclDataBuffer> self_kv_buffer;
    self_kv_buffer.reserve(self_kv_ptr_.size());
    for (auto &p : self_kv_ptr_) {
      AclDataBuffer tmp_buffer(p, n_text_ctx_ * n_text_state_ * sizeof(float));
      self_kv_buffer.push_back(std::move(tmp_buffer));

      input_dataset.AddBuffer(self_kv_buffer.back());
    }

    std::vector<AclDataBuffer> cross_kv_buffer;
    cross_kv_buffer.reserve(cross_kv_ptr_.size());
    for (auto &p : cross_kv_ptr_) {
      AclDataBuffer tmp_buffer(p,
                               num_out_frames_ * n_text_state_ * sizeof(float));
      cross_kv_buffer.push_back(std::move(tmp_buffer));

      input_dataset.AddBuffer(cross_kv_buffer.back());
    }

    AclDataBuffer offset_buf(offset_ptr_, sizeof(int32_t));
    input_dataset.AddBuffer(offset_buf);

    AclDataBuffer mask_buf(mask_ptr_, n_text_ctx_ * sizeof(int32_t));
    input_dataset.AddBuffer(mask_buf);

    AclMdlDataset output_dataset;

    AclDataBuffer logits_buf(logits_ptr_, vocab_size_ * sizeof(float));
    output_dataset.AddBuffer(logits_buf);

    std::vector<AclDataBuffer> delta_kv_buffer;
    delta_kv_buffer.reserve(delta_kv_ptr_.size());
    for (auto &p : delta_kv_ptr_) {
      AclDataBuffer tmp_buffer(p, n_text_state_ * sizeof(float));
      delta_kv_buffer.push_back(std::move(tmp_buffer));

      output_dataset.AddBuffer(delta_kv_buffer.back());
    }

    ret = aclmdlExecute(*decoder_model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlExecute");

    UpdateSelfKvCache();

    ret = aclrtMemcpy(logits_cpu_.data(), logits_cpu_.size() * sizeof(float),
                      logits_ptr_, logits_cpu_.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);

    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    return MaxElementIndex(logits_cpu_.data(), logits_cpu_.size());
  }

  void UpdateSelfKvCache() {
    int32_t offset = token_offset_mask_cpu_[1];
    for (int32_t i = 0; i < n_text_layer_ * 2; ++i) {
      const float *src = delta_kv_ptr_[i];
      float *dst = self_kv_ptr_[i] + offset * n_text_state_;

      auto ret = aclrtMemcpy(dst, n_text_state_ * sizeof(float), src,
                             n_text_state_ * sizeof(float),
                             ACL_MEMCPY_DEVICE_TO_DEVICE);
      SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");
    }
  }

  void PreInit() {
    int32_t device_id = 0;
    aclError ret = aclrtSetDevice(device_id);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclrtSetDevice with device id: %d", device_id);

    context_ = std::make_unique<AclContext>(device_id);

    ret = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");
  }

  void PostInit() {
    PostInitEncoder();
    PostInitDecoder();
    Preallocate();
    InitSotSequence();
  }

  void InitSotSequence() {
    switch (model_type_) {
      case WhisperModelType::TinyEn:
        // fallthrough
      case WhisperModelType::BaseEn:
        // fallthrough
      case WhisperModelType::SmallEn:
        // fallthrough
      case WhisperModelType::MediumEn:
        // fallthrough
        // <|startoftranscript|><|notimestamps|>
        sot_sequence_ = {50257, 50362};
        eot_ = 50256;
        break;
      case WhisperModelType::Tiny:
      case WhisperModelType::Base:
        // fallthrough
      case WhisperModelType::Small:
        // fallthrough
      case WhisperModelType::Medium:
        // fallthrough
      case WhisperModelType::Large:
        // <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        sot_sequence_ = {50258, 50259, 50359, 50363};
        eot_ = 50257;
        break;
      default:
        SHERPA_ONNX_LOGE("Unsupported model type: '%s'",
                         ToString(model_type_).c_str());
        SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      std::ostringstream os;
      os << "sot_sequence: ";
      for (auto i : sot_sequence_) {
        os << i << " ";
      }
      os << "\n";
      os << "eot: " << eot_ << "\n";
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }
  }

  void Preallocate() {
    // Allocate a single big block.
    int32_t total = 0;

    // features: (1, feat_dim_, num_frames_)
    total += num_frames_ * feat_dim_ * sizeof(float);
    // token: (1,)
    total += sizeof(int32_t);
    // offset: (1,)
    total += sizeof(int32_t);

    // mask: (1, n_text_ctx_)
    total += n_text_ctx_ * sizeof(int32_t);

    // logits: (1, 1, vocab_size_)
    total += vocab_size_ * sizeof(float);

    // cross_kv: n_text_layer_ * 2 * (num_out_frames_, n_text_state_)

    total +=
        n_text_layer_ * 2 * num_out_frames_ * n_text_state_ * sizeof(float);

    // self_kv: n_text_layer_ * 2 * (n_text_ctx_, n_text_state_)
    total += n_text_layer_ * 2 * n_text_ctx_ * n_text_state_ * sizeof(float);

    // delta_kv: n_text_layer_ * 2 * (1, 1, n_text_state_)
    total += n_text_layer_ * 2 * n_text_state_ * sizeof(float);

    ptr_ = std::make_unique<AclDevicePtr>(total);
    float *start = ptr_->Get<float>();
    int32_t *start_int32 = ptr_->Get<int32_t>();
    int32_t offset = 0;

    // (1, feat_dim_, num_frames_)
    features_ptr_ = start + offset;
    offset += feat_dim_ * num_frames_;  // in float or in int32_t, not in bytes

    // make sure token,offset,mask are contiguous in device memory

    // (1,)
    token_ptr_ = start_int32 + offset;
    offset += 1;

    // (1,)
    offset_ptr_ = start_int32 + offset;
    offset += 1;

    // (1, n_text_ctx_)
    mask_ptr_ = start_int32 + offset;
    offset += n_text_ctx_;

    // (1, 1, vocab_size_)
    logits_ptr_ = start + offset;
    offset += vocab_size_;

    // (1, num_frames_, n_text_state_)
    cross_kv_ptr_.reserve(n_text_layer_ * 2);
    for (int32_t i = 0; i < n_text_layer_ * 2; ++i) {
      auto p = start + offset;
      offset += num_out_frames_ * n_text_state_;
      cross_kv_ptr_.push_back(std::move(p));
    }

    // (1, n_text_ctx_, n_text_state_)
    self_kv_ptr_.reserve(n_text_layer_ * 2);
    for (int32_t i = 0; i < n_text_layer_ * 2; ++i) {
      auto p = start + offset;
      offset += n_text_ctx_ * n_text_state_;
      self_kv_ptr_.push_back(std::move(p));
    }

    // (1, 1, n_text_state_)
    delta_kv_ptr_.reserve(n_text_layer_ * 2);
    for (int32_t i = 0; i < n_text_layer_ * 2; ++i) {
      auto p = start + offset;
      offset += n_text_state_;
      delta_kv_ptr_.push_back(std::move(p));
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Allocated %d bytes, or %.3f MB", total,
                       total / 1024. / 1024.);
    }
  }

  void PostInitEncoder() {
    const std::vector<std::string> &names = encoder_model_->GetInputNames();
    model_type_ = ParseWhisperModelFromString(names[0]);
    if (config_.debug) {
      SHERPA_ONNX_LOGE("model type: %s", ToString(model_type_).c_str());
    }

    const std::vector<std::vector<int64_t>> &input_shapes =
        encoder_model_->GetInputShapes();

    const auto &mel_shape = input_shapes[0];
    if (mel_shape[0] != 1) {
      SHERPA_ONNX_LOGE("It supports only batch size == 1. Given: %d",
                       static_cast<int32_t>(mel_shape[0]));
      SHERPA_ONNX_EXIT(-1);
    }

    feat_dim_ = mel_shape[1];
    num_frames_ = mel_shape[2];

    const std::vector<std::vector<int64_t>> &output_shapes =
        encoder_model_->GetOutputShapes();

    n_text_layer_ = output_shapes.size() / 2;

    num_out_frames_ = output_shapes[0][1];
    n_text_state_ = output_shapes[0].back();

    if (config_.debug) {
      SHERPA_ONNX_LOGE("feat_dim_: %d", feat_dim_);
      SHERPA_ONNX_LOGE("num_frames_: %d", num_frames_);
      SHERPA_ONNX_LOGE("num_out_frames_: %d", num_out_frames_);
      SHERPA_ONNX_LOGE("n_text_layer_: %d", n_text_layer_);
      SHERPA_ONNX_LOGE("n_text_state_: %d", n_text_state_);
    }
  }

  void PostInitDecoder() {
    const std::vector<std::vector<int64_t>> &input_shapes =
        decoder_model_->GetInputShapes();
    // tokens, self_kv, cross_kv, offset, mask
    int32_t expected_num_inputs = 1 + 2 * n_text_layer_ + 2 * n_text_layer_ + 2;
    if (input_shapes.size() != expected_num_inputs) {
      SHERPA_ONNX_LOGE("Expect %d inputs. Actual: %d", expected_num_inputs,
                       static_cast<int32_t>(input_shapes.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &s = input_shapes[1];
    if (s[0] != 1) {
      SHERPA_ONNX_LOGE("Support only batch size 1. Given: %d",
                       static_cast<int32_t>(s[0]));
      SHERPA_ONNX_EXIT(-1);
    }

    n_text_ctx_ = s[1];
    token_offset_mask_cpu_.resize(1 + 1 + n_text_ctx_);

    if (s[2] != n_text_state_) {
      SHERPA_ONNX_LOGE("Expect n_text_state_ %d. Given: %d", n_text_state_,
                       static_cast<int32_t>(s[2]));
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("n_text_ctx_: %d", n_text_ctx_);
    }

    const std::vector<std::vector<int64_t>> &output_shapes =
        decoder_model_->GetOutputShapes();

    vocab_size_ = output_shapes[0].back();
    logits_cpu_.resize(vocab_size_);

    if (config_.debug) {
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
    }
  }

  void InitEncoder(const std::string &filename) {
    encoder_model_ = std::make_unique<AclModel>(filename);
    if (config_.debug) {
      auto s = encoder_model_->GetInfo();

      SHERPA_ONNX_LOGE("----encoder----\n%s\n", s.c_str());
    }
  }

  void InitDecoder(const std::string &filename) {
    decoder_model_ = std::make_unique<AclModel>(filename);
    if (config_.debug) {
      auto s = decoder_model_->GetInfo();

      SHERPA_ONNX_LOGE("----decoder----\n%s\n", s.c_str());
    }
  }

 private:
  std::mutex mutex_;
  Acl acl_;

  std::unique_ptr<AclContext> context_;

  std::unique_ptr<AclModel> encoder_model_;
  std::unique_ptr<AclModel> decoder_model_;

  OfflineModelConfig config_;

  // tiny, tiny.en, base.en, base, etc
  WhisperModelType model_type_;
  int32_t feat_dim_ = 0;
  int32_t num_frames_ = 0;
  int32_t num_out_frames_ = 0;
  int32_t n_text_layer_ = 0;
  int32_t n_text_ctx_ = 0;
  int32_t n_text_state_ = 0;
  int32_t vocab_size_ = 0;

  std::unique_ptr<AclDevicePtr> ptr_;

  // All of the following raw pointers will point to some already allocated
  // device memory. No need to free them.
  float *features_ptr_ = nullptr;
  int32_t *token_ptr_ = nullptr;
  int32_t *offset_ptr_ = nullptr;
  int32_t *mask_ptr_ = nullptr;
  float *logits_ptr_ = nullptr;

  std::vector<float *> cross_kv_ptr_;
  std::vector<float *> self_kv_ptr_;
  std::vector<float *> delta_kv_ptr_;

  std::vector<int32_t> token_offset_mask_cpu_;
  std::vector<float> logits_cpu_;

  std::vector<int32_t> sot_sequence_;
  int32_t eot_ = 0;
};

OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineWhisperModelAscend::~OfflineWhisperModelAscend() = default;

OfflineWhisperDecoderResult OfflineWhisperModelAscend::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineWhisperModelAscend::FeatureDim() const {
  return impl_->FeatureDim();
}

#if __ANDROID_API__ >= 9
template OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
