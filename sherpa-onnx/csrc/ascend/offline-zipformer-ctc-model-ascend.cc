// sherpa-onnx/csrc/ascend/offline-zipformer-ctc-model-ascend.cc
//
// Copyright (c)  2025  Xiaomi Corporation

// References:
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/appdevgapi/aclcppdevg_03_0298.html
#include "sherpa-onnx/csrc/ascend/offline-zipformer-ctc-model-ascend.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>  // NOLINT
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

namespace sherpa_onnx {

class OfflineZipformerCtcModelAscend::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    PreInit();
    InitModel(config_.zipformer_ctc.model);
    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    PreInit();
    {
      auto buf = ReadFile(mgr, config_.zipformer_ctc.model);
      InitModel(buf.data(), buf.size());
    }
    PostInit();
  }

  std::vector<float> Run(std::vector<float> features) {
    // TODO(fangjun): Support multi clients
    std::lock_guard<std::mutex> lock(mutex_);

    int32_t num_frames = features.size() / feat_dim_;

    if (num_frames != max_num_frames_) {
      if (num_frames > max_num_frames_) {
        SHERPA_ONNX_LOGE(
            "Number of input frames %d is too large. Truncate it to %d frames.",
            num_frames, max_num_frames_);

        SHERPA_ONNX_LOGE(
            "Recognition result may be truncated/incomplete. Please select a "
            "model accepting longer audios.");
      }

      features.resize(max_num_frames_ * feat_dim_);

      num_frames = max_num_frames_;
    }

    aclError ret =
        aclrtMemcpy(*x_ptr_, features.size() * sizeof(float), features.data(),
                    features.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    AclMdlDataset input_dataset;
    AclDataBuffer x_buf(*x_ptr_, features.size() * sizeof(float));
    input_dataset.AddBuffer(x_buf);

    AclMdlDataset output_dataset;

    AclDataBuffer logits_buf(*log_probs_ptr_,
                             num_output_frames_ * vocab_size_ * sizeof(float));
    output_dataset.AddBuffer(logits_buf);

    ret = aclmdlExecute(*model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlExecute");

    std::vector<float> log_probs(num_output_frames_ * vocab_size_);
    ret = aclrtMemcpy(
        log_probs.data(), num_output_frames_ * vocab_size_ * sizeof(float),
        *log_probs_ptr_, num_output_frames_ * vocab_size_ * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    return log_probs;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

 private:
  void InitModel(const std::string &filename) {
    model_ = std::make_unique<AclModel>(filename);
    if (config_.debug) {
      auto s = model_->GetInfo();
      SHERPA_ONNX_LOGE("%s", s.c_str());
    }
  }

  void InitModel(void *data, size_t size) {
    model_ = std::make_unique<AclModel>(data, size);
    if (config_.debug) {
      auto s = model_->GetInfo();
      SHERPA_ONNX_LOGE("%s", s.c_str());
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
    auto in_shape = model_->GetInputShapes()[0];

    max_num_frames_ = in_shape[1];
    feat_dim_ = in_shape[2];

    auto out_shape = model_->GetOutputShapes()[0];

    num_output_frames_ = out_shape[1];
    vocab_size_ = out_shape[2];

    subsampling_factor_ = max_num_frames_ / out_shape[1];
    if (config_.debug) {
      SHERPA_ONNX_LOGE("max_num_frames: %d", max_num_frames_);
      SHERPA_ONNX_LOGE("feat_dim: %d", feat_dim_);
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
      SHERPA_ONNX_LOGE("subsampling_factor: %d", subsampling_factor_);
    }

    Preallocate();
  }

  void Preallocate() {
    x_ptr_ = std::make_unique<AclDevicePtr>(max_num_frames_ * feat_dim_ *
                                            sizeof(float));

    log_probs_ptr_ = std::make_unique<AclDevicePtr>(
        num_output_frames_ * vocab_size_ * sizeof(float));
  }

 private:
  std::mutex mutex_;
  Acl acl_;

  std::unique_ptr<AclContext> context_;

  OfflineModelConfig config_;

  std::unique_ptr<AclModel> model_;
  int32_t vocab_size_ = 0;
  int32_t max_num_frames_ = 0;
  int32_t num_output_frames_ = 0;
  int32_t feat_dim_ = 0;
  int32_t subsampling_factor_ = 0;

  std::unique_ptr<AclDevicePtr> x_ptr_;
  std::unique_ptr<AclDevicePtr> log_probs_ptr_;
};

OfflineZipformerCtcModelAscend::OfflineZipformerCtcModelAscend(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineZipformerCtcModelAscend::OfflineZipformerCtcModelAscend(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineZipformerCtcModelAscend::~OfflineZipformerCtcModelAscend() = default;

std::vector<float> OfflineZipformerCtcModelAscend::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineZipformerCtcModelAscend::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OfflineZipformerCtcModelAscend::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

#if __ANDROID_API__ >= 9
template OfflineZipformerCtcModelAscend::OfflineZipformerCtcModelAscend(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineZipformerCtcModelAscend::OfflineZipformerCtcModelAscend(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
