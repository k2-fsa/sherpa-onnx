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
static std::vector<float> CreateCausalMask(int32_t n, int32_t capacity) {
  std::vector<float> mask(capacity, 1);
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

  std::vector<int32_t> Run(std::vector<float> features) {
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
    }

    features.resize(num_frames_ * feat_dim_);

    // (num_frames_, feat_dim_) -> (feat_dim_, num_frames_)
    features = Transpose(features.data(), num_frames_, feat_dim_);

    SHERPA_ONNX_LOGE("features size: %d. %dx%d", (int)features.size(),
                     (int)features.size() / feat_dim_, feat_dim_);
    RunEncoder(std::move(features));
    SHERPA_ONNX_LOGE("run encoder done!");

    return {};
  }

  int32_t FeatureDim() const { return feat_dim_; }

 private:
  void RunEncoder(std::vector<float> features) {
    aclError ret = aclrtMemcpy(*features_ptr_, features.size() * sizeof(float),
                               features.data(), features.size() * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE);

    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    AclMdlDataset input_dataset;
    AclDataBuffer features_buf(*features_ptr_, features.size() * sizeof(float));
    input_dataset.AddBuffer(features_buf);

    AclMdlDataset output_dataset;

    std::vector<AclDataBuffer> cross_kv_buffer;
    cross_kv_buffer.reserve(cross_kv_ptr_.size());
    for (auto &p : cross_kv_ptr_) {
      AclDataBuffer tmp_buffer(*p,
                               num_out_frames_ * n_text_state_ * sizeof(float));
      cross_kv_buffer.push_back(std::move(tmp_buffer));

      output_dataset.AddBuffer(cross_kv_buffer.back());
    }

    ret = aclmdlExecute(*encoder_model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlExecute");
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
  }

  void Preallocate() {
    // TODO(fangjun): Allocate a single big block.
    int32_t total = 0;
    features_ptr_ =
        std::make_unique<AclDevicePtr>(num_frames_ * feat_dim_ * sizeof(float));

    total += num_frames_ * feat_dim_ * sizeof(float);

    cross_kv_ptr_.reserve(n_text_layer_ * 2);
    for (int32_t i = 0; i < n_text_layer_ * 2; ++i) {
      auto p = std::make_unique<AclDevicePtr>(num_out_frames_ * n_text_state_ *
                                              sizeof(float));
      cross_kv_ptr_.push_back(std::move(p));
      total += num_out_frames_ * n_text_state_ * sizeof(float);
    }

    self_kv_ptr_.reserve(n_text_layer_ * 2);
    for (int32_t i = 0; i < n_text_layer_ * 2; ++i) {
      auto p = std::make_unique<AclDevicePtr>(n_text_ctx_ * n_text_state_ *
                                              sizeof(float));
      self_kv_ptr_.push_back(std::move(p));
      total += n_text_ctx_ * n_text_state_ * sizeof(float);
    }

    delta_kv_ptr_.reserve(n_text_layer_ * 2);
    for (int32_t i = 0; i < n_text_layer_ * 2; ++i) {
      auto p = std::make_unique<AclDevicePtr>(n_text_state_ * sizeof(float));
      delta_kv_ptr_.push_back(std::move(p));
      total += n_text_state_ * sizeof(float);
    }
    SHERPA_ONNX_LOGE("Allocated %.3f MB", total / 1024. / 1024.);
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
    if (s[2] != n_text_state_) {
      SHERPA_ONNX_LOGE("Expect n_text_state_ %d. Given: %d", n_text_state_,
                       static_cast<int32_t>(s[2]));
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("n_text_ctx_: %d", n_text_ctx_);
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

  std::unique_ptr<AclDevicePtr> features_ptr_;

  std::vector<std::unique_ptr<AclDevicePtr>> cross_kv_ptr_;
  std::vector<std::unique_ptr<AclDevicePtr>> self_kv_ptr_;
  std::vector<std::unique_ptr<AclDevicePtr>> delta_kv_ptr_;
};

OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineWhisperModelAscend::~OfflineWhisperModelAscend() = default;

std::vector<int32_t> OfflineWhisperModelAscend::Run(
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
