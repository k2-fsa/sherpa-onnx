// sherpa-onnx/csrc/ascend/offline-sense-voice-model-ascend.cc
//
// Copyright (c)  2025  Xiaomi Corporation

// References:
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/appdevgapi/aclcppdevg_03_0298.html
#include "sherpa-onnx/csrc/ascend/offline-sense-voice-model-ascend.h"

#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "sherpa-onnx/csrc/ascend/macros.h"
#include "sherpa-onnx/csrc/ascend/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"

namespace sherpa_onnx {

std::vector<float> LoadFeatures() {
  FILE *fp = fopen("./features.bin", "rb");
  fseek(fp, 0, SEEK_END);
  int32_t n = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  printf("n: %d\n", n);

  std::vector<float> d(n / sizeof(float));
  fread(d.data(), sizeof(float), n, fp);
  fclose(fp);

  return d;
}

class OfflineSenseVoiceModelAscend::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    InitModel(config_.sense_voice.model);
    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(mgr, config_.sense_voice.model);
      InitModel(buf.data(), buf.size());
    }
    PostInit();
  }

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  void Test() {
    std::vector<float> features = LoadFeatures();
    int32_t num_frames = features.size() / 560;
    std::cout << "num_frames: " << num_frames << "\n";

    aclError ret =
        aclrtMemcpy(*x_ptr_, features.size() * sizeof(float), features.data(),
                    features.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    std::array<int32_t, 4> prompt_array = {0, 1, 2, 15};
    ret = aclrtMemcpy(*prompt_ptr_, prompt_ptr_->Size(), prompt_array.data(),
                      prompt_ptr_->Size(), ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    AclMdlDataset input_dataset;
    AclDataBuffer x_buf(*x_ptr_, features.size() * sizeof(float));
    input_dataset.AddBuffer(x_buf);

    AclDataBuffer prompt_buf(*prompt_ptr_, prompt_ptr_->Size());
    input_dataset.AddBuffer(prompt_buf);

    // 动态Shape输入（设置Shape范围）
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/appdevg/acldevg/aclcppdevg_000044.html

    std::array<int64_t, 3> x_shape = {1, 93, 560};
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

    const float *p = logits.data();
    int32_t blank = 0;
    int32_t prev_id = -1;
    std::vector<int32_t> tokens;

    for (int32_t t = 0; t != num_frames; ++t) {
      auto y = static_cast<int64_t>(std::distance(
          static_cast<const float *>(p),
          std::max_element(static_cast<const float *>(p),
                           static_cast<const float *>(p) + vocab_size_)));
      p += vocab_size_;

      if (y != 0 && y != prev_id) {
        tokens.push_back(y);
      }
      prev_id = y;
    }  // for (int32_t t = 0; ...)

    for (auto t : tokens) {
      std::cout << t << ", ";
    }
    std::cout << "\n";
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

  void PostInit() {
    vocab_size_ = model_->GetOutputShapes()[0].back();

    Preallocate();
  }

  void Preallocate() {
    max_num_frames_ = (10 * 100 - 7) / 6 + 1;
    x_ptr_ = std::make_unique<AclDevicePtr>(max_num_frames_ * feat_dim_ *
                                            sizeof(float));

    prompt_ptr_ = std::make_unique<AclDevicePtr>(4 * sizeof(int32_t));

    logits_ptr_ = std::make_unique<AclDevicePtr>((max_num_frames_ + 4) *
                                                 vocab_size_ * sizeof(float));
  }

 private:
  OfflineModelConfig config_;
  OfflineSenseVoiceModelMetaData meta_data_;

  std::unique_ptr<AclModel> model_;
  int32_t vocab_size_ = 0;
  int32_t max_num_frames_ = 0;
  int32_t feat_dim_ = 560;

  std::unique_ptr<AclDevicePtr> x_ptr_;
  std::unique_ptr<AclDevicePtr> prompt_ptr_;
  std::unique_ptr<AclDevicePtr> logits_ptr_;
};

OfflineSenseVoiceModelAscend::OfflineSenseVoiceModelAscend(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineSenseVoiceModelAscend::~OfflineSenseVoiceModelAscend() = default;

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
