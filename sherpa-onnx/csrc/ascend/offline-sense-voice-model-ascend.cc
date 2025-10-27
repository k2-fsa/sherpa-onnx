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
  Impl() {
    std::vector<float> features = LoadFeatures();
    int32_t num_frames = features.size() / 560;
    std::cout << "num_frames: " << num_frames << "\n";

    std::string filename = "./model.om";
    model_ = std::make_unique<AclModel>(filename);
    auto s = model_->GetInfo();
    SHERPA_ONNX_LOGE("%s", s.c_str());

    AclDevicePtr x(features.size() * sizeof(float));

    aclError ret = aclrtMemcpy(x.Get(), x.Size(), features.data(), x.Size(),
                               ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    aclmdlDataset *input_dataset = aclmdlCreateDataset();

    aclDataBuffer *x_buf = aclCreateDataBuffer(x.Get(), x.Size());
    ret = aclmdlAddDatasetBuffer(input_dataset, x_buf);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlAddDatasetBuffer");

    std::array<int32_t, 4> prompt_array = {0, 1, 2, 15};
    AclDevicePtr prompt(prompt_array.size() * sizeof(int32_t));

    ret = aclrtMemcpy(prompt.Get(), prompt.Size(), prompt_array.data(),
                      prompt.Size(), ACL_MEMCPY_HOST_TO_DEVICE);

    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    aclDataBuffer *prompt_buf =
        aclCreateDataBuffer(prompt.Get(), prompt.Size());
    ret = aclmdlAddDatasetBuffer(input_dataset, prompt_buf);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlAddDatasetBuffer");

    // 动态Shape输入（设置Shape范围）
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/appdevg/acldevg/aclcppdevg_000044.html

    std::array<int64_t, 3> x_shape = {1, 93, 560};
    aclTensorDesc *x_desc = aclCreateTensorDesc(ACL_FLOAT, x_shape.size(),
                                                x_shape.data(), ACL_FORMAT_ND);
    ret = aclmdlSetDatasetTensorDesc(input_dataset, x_desc, 0);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclmdlSetDatasetTensorDesc for input 0");

    std::array<int64_t, 1> prompt_shape = {4};
    aclTensorDesc *prompt_desc = aclCreateTensorDesc(
        ACL_INT32, prompt_shape.size(), prompt_shape.data(), ACL_FORMAT_ND);
    ret = aclmdlSetDatasetTensorDesc(input_dataset, prompt_desc, 1);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclmdlSetDatasetTensorDesc for input 2");

    aclmdlDataset *output_dataset = aclmdlCreateDataset();

    aclDataBuffer *logits_buf = aclCreateDataBuffer(nullptr, 0);
    ret = aclmdlAddDatasetBuffer(output_dataset, logits_buf);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclmdlAddDatasetBuffer for output 0");

    ret = aclmdlExecute(model_->Get(), input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlExecute");

    size_t logits_size = aclGetDataBufferSizeV2(logits_buf);
    SHERPA_ONNX_LOGE("logits size %d, num_frames: %d\n", (int)logits_size,
                     (int)(logits_size / 25055 / sizeof(float)));

    void *data = aclGetDataBufferAddr(logits_buf);

    std::vector<float> logits(logits_size / sizeof(float));
    ret = aclrtMemcpy(logits.data(), logits_size, data, logits_size,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    ret = aclrtFree(data);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtFree");

    int32_t vocab_size = 25055;
    int32_t num_out_frames = logits.size() / 25055;

    const float *p = logits.data();
    int32_t blank = 0;
    int32_t prev_id = -1;
    std::vector<int32_t> tokens;

    for (int32_t t = 0; t != num_out_frames; ++t) {
      auto y = static_cast<int64_t>(std::distance(
          static_cast<const float *>(p),
          std::max_element(static_cast<const float *>(p),
                           static_cast<const float *>(p) + vocab_size)));
      p += vocab_size;

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
  std::unique_ptr<AclModel> model_;
};

OfflineSenseVoiceModelAscend::OfflineSenseVoiceModelAscend()
    : impl_(std::make_unique<Impl>()) {}

OfflineSenseVoiceModelAscend::~OfflineSenseVoiceModelAscend() = default;

}  // namespace sherpa_onnx
