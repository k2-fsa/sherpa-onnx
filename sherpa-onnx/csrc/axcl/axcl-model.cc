// sherpa-onnx/csrc/axcl/axcl-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axcl/axcl-model.h"

#include <memory>
#include <string>
#include <vector>

#include "axcl.h"  // NOLINT
#include "sherpa-onnx/csrc/axcl/axcl-engine-guard.h"
#include "sherpa-onnx/csrc/axcl/axcl-manager.h"
#include "sherpa-onnx/csrc/axcl/utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

/*
Initialization step:

1. AxclInit()
2. set device
3. init engine
4. axclrtEngineLoadFromMem or axclrtEngineLoadFromFile
 */

class AxclModel::Impl {
 public:
  Impl(const std::string &filename, int32_t device_id) {
    if (!SetDevice(device_id)) {
      return;
    }

    InitEngine();

    axclError ret = axclrtEngineLoadFromFile(filename, &model_id_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE(
          "Failed to call axclrtEngineLoadFromFile() with file: %s. Return "
          "code is: %d",
          filename.c_str(), static_cast<int32_t>(0));
      SHERPA_ONNX_EXIT(-1);
    }

    model_loaded_ = true;
  }

  Impl(const void *cpu_buf, size_t buf_len_in_bytes, int32_t device_id) {
    if (!SetDevice(device_id)) {
      return;
    }

    InitEngine();

    {
      AxclDevicePtr device_ptr(buf_len_in_bytes, AXCL_MEM_MALLOC_NORMAL_ONLY);
      auto ret = axclrtMemcpy(device_ptr, cpu_buf, buf_len_in_bytes,
                              AXCL_MEMCPY_HOST_TO_DEVICE);
      if (ret != 0) {
        SHERPA_ONNX_LOGE("Failed to call axclrtMemcpy(). Return code is: %d",
                         static_cast<int32_t>(ret));
        return;
      }

      ret = axclrtEngineLoadFromMem(device_ptr, buf_len_in_bytes, &model_id_);
      if (ret != 0) {
        SHERPA_ONNX_LOGE(
            "Failed to call axclrtEngineLoadFromMem(). Return code is: %d",
            static_cast<int32_t>(ret));
        return;
      }

      model_loaded_ = true;
    }
  }

  ~Impl() {
    if (model_loaded_) {
      axclError ret = axclrtEngineUnload(model_id_);

      if (ret != 0) {
        SHERPA_ONNX_LOGE(
            "Failed to call axclrtEngineUnload(). Return code is: %d",
            static_cast<int32_t>(ret));
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

 private:
  bool SetDevice(int32_t device_id) {
    axclrtDeviceList lst;
    auto ret = axclrtGetDeviceList(&lst);
    if (ret != 0) {
      SHERPA_ONNX_LOGE(
          "Failed to call axclrtGetDeviceList(). Return code is: %d",
          static_cast<int32_t>(ret));
      return false;
    }

    if (lst.num == 0) {
      SHERPA_ONNX_LOGE("Found 0 device.");
      return false;
    }

    // device_id counts from 0
    if (device_id < 0 || device_id >= lst.num) {
      SHERPA_ONNX_LOGE("Invalid device_id: %d. Validate range: 0-%d", device_id,
                       0, lst.num - 1);
      return false;
    }

    ret = axclrtSetDevice(lst.devices[device_id]);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to call axclrtSetDevice(). Return code is: %d",
                       static_cast<int32_t>(ret));
      return false;
    }

    return true;
  }

  void InitEngine() { engine_guard_ = std::make_unique<AxclEngineGuard>(); }

 private:
  AxclManager manager_;
  std::unique_ptr<AxclEngineGuard> engine_guard_;

  bool model_loaded_ = false;
  uint64_t model_id_ = 0;
};

AxclModel::AxclModel(const std::string &filename, int32_t device_id /*= 0*/)
    : impl_(std::make_unique<Impl>(filename, device_id)) {}

AxclModel::AxclModel(const void *cpu_buf, size_t buf_len_in_bytes,
                     int32_t device_id /*= 0*/)
    : impl_(std::make_unique<Impl>(cpu_buf, buf_len_in_bytes, device_id)) {}

const std::vector<std::string> &AxclModel::InputTensorNames() const {
  return impl_->InputTensorNames();
}
const std::vector<std::string> &AclModel::OutputTensorNames() const {
  return impl_->OutputTensorNames();
}

std::vector<int32_t> AclModel::TensorShape(const std::string &name) const {
  return impl_->TensorShape(name);
}

int32_t AclModel::TensorSizeInBytes(const std::string &name) const {
  return impl_->TensorSizeInBytes(name);
}

bool AclModel::HasTensor(const std::string &name) const {
  return impl_->HasTensor(name);
}

bool AclModel::SetInputTensorData(const std::string &name, const float *p,
                                  int32_t n) const {
  return impl_->SetInputTensorData(name, p, n);
}

bool AclModel::SetInputTensorData(const std::string &name, const int32_t *p,
                                  int32_t n) const {
  return impl_->SetInputTensorData(name, p, n);
}

std::vector<float> AclModel::GetOutputTensorData(
    const std::string &name) const {
  return impl_->GetOutputTensorData(name);
}

bool AclModel::Run() const { return impl_->Run(); }

bool AclModel::IsInitialized() const { return impl_->IsInitialized(); }

}  // namespace sherpa_onnx
