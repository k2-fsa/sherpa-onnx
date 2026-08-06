// sherpa-onnx/csrc/ascend/offline-sense-voice-model-ascend.cc
//
// Copyright (c)  2025  Xiaomi Corporation

// References:
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/appdevgapi/aclcppdevg_03_0298.html
#include "sherpa-onnx/csrc/ascend/offline-sense-voice-model-ascend.h"

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
#include "sherpa-onnx/csrc/lfr.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelAscend::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    PreInit();
    InitModel(config_.sense_voice.model);
    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    PreInit();
    {
      auto buf = ReadFile(mgr, config_.sense_voice.model);
      InitModel(buf.data(), buf.size());
    }
    PostInit();
  }

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) {
    // TODO(fangjun): Support multi clients
    std::lock_guard<std::mutex> lock(mutex_);

    aclError ret_set_ctx = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret_set_ctx,
                            "Failed to call aclrtSetCurrentContext");

    features = ApplyLFR(std::move(features));
    if (features.empty()) {
      return {};
    }

    int32_t num_frames = features.size() / kLfrOutputDim;

    aclError ret =
        aclrtMemcpy(*x_ptr_, features.size() * sizeof(float), features.data(),
                    features.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    std::array<int32_t, 4> prompt_array{language, 1, 2, text_norm};
    ret = aclrtMemcpy(*prompt_ptr_, prompt_ptr_->Size(), prompt_array.data(),
                      prompt_ptr_->Size(), ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    AclMdlDataset input_dataset;
    AclDataBuffer x_buf(*x_ptr_, features.size() * sizeof(float));
    input_dataset.AddBuffer(x_buf);

    AclDataBuffer prompt_buf(*prompt_ptr_, prompt_ptr_->Size());
    input_dataset.AddBuffer(prompt_buf);

    // dynamic shape input
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/appdevg/acldevg/aclcppdevg_000044.html

    std::array<int64_t, 3> x_shape = {1, num_frames, kLfrOutputDim};
    AclTensorDesc x_desc(ACL_FLOAT, x_shape.size(), x_shape.data(),
                         ACL_FORMAT_ND);
    input_dataset.SetTensorDesc(x_desc, 0);

    std::array<int64_t, 1> prompt_shape = {4};
    AclTensorDesc prompt_desc(ACL_INT32, prompt_shape.size(),
                              prompt_shape.data(), ACL_FORMAT_ND);
    input_dataset.SetTensorDesc(prompt_desc, 1);

    AclMdlDataset output_dataset;

    AclDataBuffer logits_buf(*logits_ptr_,
                             num_frames * vocab_size_ * sizeof(float));
    output_dataset.AddBuffer(logits_buf);

    ret = aclmdlExecute(*model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlExecute");

    std::vector<float> logits(num_frames * vocab_size_);
    ret = aclrtMemcpy(logits.data(), num_frames * vocab_size_ * sizeof(float),
                      *logits_ptr_, num_frames * vocab_size_ * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    return logits;
  }

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
    vocab_size_ = model_->GetOutputShapes()[0].back();

    Preallocate();
  }

  void Preallocate() {
    // max 30 seconds
    max_num_frames_ = (30 * 100 + kLfrWindowShift - 1) / kLfrWindowShift;
    x_ptr_ = std::make_unique<AclDevicePtr>(max_num_frames_ * kLfrOutputDim *
                                            sizeof(float));

    prompt_ptr_ = std::make_unique<AclDevicePtr>(4 * sizeof(int32_t));

    logits_ptr_ = std::make_unique<AclDevicePtr>((max_num_frames_ + 4) *
                                                 vocab_size_ * sizeof(float));
  }

  std::vector<float> ApplyLFR(const std::vector<float> &in) const {
    std::vector<float> out =
        ApplyLfr(in, kLfrInputDim, kLfrWindowSize, kLfrWindowShift);
    int32_t out_num_frames = static_cast<int32_t>(out.size() / kLfrOutputDim);

    if (out_num_frames > max_num_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          out_num_frames, max_num_frames_);

      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios.");

      out_num_frames = max_num_frames_;
      out.resize(static_cast<size_t>(out_num_frames) * kLfrOutputDim);
    }

    return out;
  }

 private:
  static constexpr int32_t kLfrInputDim = 80;
  static constexpr int32_t kLfrWindowSize = 7;
  static constexpr int32_t kLfrWindowShift = 6;
  static constexpr int32_t kLfrOutputDim = kLfrInputDim * kLfrWindowSize;

  std::mutex mutex_;
  Acl acl_;

  std::unique_ptr<AclContext> context_;

  OfflineModelConfig config_;
  OfflineSenseVoiceModelMetaData meta_data_;

  std::unique_ptr<AclModel> model_;
  int32_t vocab_size_ = 0;
  int32_t max_num_frames_ = 0;

  std::unique_ptr<AclDevicePtr> x_ptr_;
  std::unique_ptr<AclDevicePtr> prompt_ptr_;
  std::unique_ptr<AclDevicePtr> logits_ptr_;
};

OfflineSenseVoiceModelAscend::OfflineSenseVoiceModelAscend(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSenseVoiceModelAscend::OfflineSenseVoiceModelAscend(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineSenseVoiceModelAscend::~OfflineSenseVoiceModelAscend() = default;

std::vector<float> OfflineSenseVoiceModelAscend::Run(
    std::vector<float> features, int32_t language, int32_t text_norm) const {
  return impl_->Run(std::move(features), language, text_norm);
}

const OfflineSenseVoiceModelMetaData &
OfflineSenseVoiceModelAscend::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

#if __ANDROID_API__ >= 9
template OfflineSenseVoiceModelAscend::OfflineSenseVoiceModelAscend(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineSenseVoiceModelAscend::OfflineSenseVoiceModelAscend(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
