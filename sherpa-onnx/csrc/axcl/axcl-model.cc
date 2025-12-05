// sherpa-onnx/csrc/axcl/axcl-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axcl/axcl-model.h"

#include <memory>
#include <string>
#include <vector>

#include "axcl.h"  // NOLINT
#include "sherpa-onnx/csrc/axcl/axcl-engine-guard.h"
#include "sherpa-onnx/csrc/axcl/axcl-engine-io-guard.h"
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
5. axclrtEngineCreateContext
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

    PostInit();
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
    }

    model_loaded_ = true;

    PostInit();
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

  const std::vector<std::string> &InputTensorNames() const {
    return input_tensor_names_;
  }
  const std::vector<std::string> &OutputTensorNames() const {
    return output_tensor_names_;
  }

  const std::vector<int32_t> &TensorShape(const std::string &name) const {
    for (size_t i = 0; i < input_tensor_names_.size(); ++i) {
      if (input_tensor_names_[i] == name) {
        return input_tensors_shapes_[i];
      }
    }

    for (size_t i = 0; i < output_tensor_names_.size(); ++i) {
      if (output_tensor_names_[i] == name) {
        return output_tensors_shapes_[i];
      }
    }

    SHERPA_ONNX_LOGE("Found no tensor with name: '%s'", name.c_str());
    return {};
  }

  int32_t TensorSizeInBytes(const std::string &name) const {
    for (size_t i = 0; i < input_tensor_names_.size(); ++i) {
      if (input_tensor_names_[i] == name) {
        return input_tensors_[i].Size();
      }
    }

    for (size_t i = 0; i < output_tensor_names_.size(); ++i) {
      if (output_tensor_names_[i] == name) {
        return output_tensors_[i].Size();
      }
    }

    SHERPA_ONNX_LOGE("Found no tensor with name: '%s'", name.c_str());
    return 0;
  }

  bool HasTensor(const std::string &name) const {
    for (size_t i = 0; i < input_tensor_names_.size(); ++i) {
      if (input_tensor_names_[i] == name) {
        return true;
      }
    }

    for (size_t i = 0; i < output_tensor_names_.size(); ++i) {
      if (output_tensor_names_[i] == name) {
        return true;
      }
    }

    return false;
  }

  template <typename T>
  bool SetInputTensorData(const std::string &name, const T *p,
                          int32_t n) const {
    for (size_t i = 0; i < input_tensor_names_.size(); ++i) {
      if (input_tensor_names_[i] == name) {
        if (n * sizeof(T) != input_tensors_[i].Size()) {
          SHERPA_ONNX_LOGE("Expected size: %zu, given: %zu",
                           input_tensors_[i].Size(), n * sizeof(T));
          return false;
        }

        auto ret = axclrtMemcpy(input_tensors_[i], p, input_tensors_[i].Size(),
                                AXCL_MEMCPY_HOST_TO_DEVICE);
        if (ret != 0) {
          SHERPA_ONNX_LOGE(
              "Failed to call axclrtMemcpy(). tensor name: '%s', return code: "
              "%d",
              name.c_str(), static_cast<int32_t>(ret));
          return false;
        }
      }
    }

    SHERPA_ONNX_LOGE("Found no tensor with name: '%s'", name.c_str());

    return false;
  }

  std::vector<float> GetOutputTensorData(const std::string &name) const {
    for (size_t i = 0; i < output_tensor_names_.size(); ++i) {
      if (output_tensor_names_[i] == name) {
        size_t bytes = output_tensors_[i].Size();
        std::vector<float> out(bytes / sizeof(float));

        auto ret = axclrtMemcpy(out.data(), output_tensors_[i], bytes,
                                AXCL_MEMCPY_DEVICE_TO_HOST);
        if (ret != 0) {
          SHERPA_ONNX_LOGE(
              "Failed to call axclrtMemcpy(). tensor name: '%s', return code: "
              "%d",
              name.c_str(), static_cast<int32_t>(ret));
          return {};
        }

        return out;
      }
    }

    SHERPA_ONNX_LOGE("Found no tensor with name: '%s'", name.c_str());

    return {};
  }

  bool Run() const {
    uint32_t group = 0;
    auto ret =
        axclrtEngineExecute(model_id_, context_id, group, engine_io_guard_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to call axclrtEngineExecute(), return code: %d",
                       static_cast<int32_t>(ret));
      return false;
    }
    return true;
  }

  bool IsInitialized() const { return model_loaded_; }

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

  void PostInit() {
    InitContext();

    axclError ret = axclrtEngineGetIOInfo(model_id_, &io_info_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE(
          "Failed to call axclrtEngineGetIOInfo(). Return code is: %d",
          static_cast<int32_t>(ret));
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t count = 0;
    ret = axclrtEngineGetShapeGroupsCount(io_info_, &count);
    if (ret != 0) {
      SHERPA_ONNX_LOGE(
          "Failed to call axclrtEngineGetShapeGroupsCount(). Return code is: "
          "%d",
          static_cast<int32_t>(ret));
      SHERPA_ONNX_EXIT(-1);
    }

    if (count != 1) {
      SHERPA_ONNX_LOGE("Only support 1 group at present. Given: %d", count);
      SHERPA_ONNX_EXIT(-1);
    }

    engine_io_guard_ = std::make_unique<AxclEngineIOGuard>(io_info_);

    InitInput();
    InitOutput();
  }

  void InitContext() {
    // Note(fangjun): No need to destroy context_id_
    auto ret = axclrtEngineCreateContext(model_id_, context_id_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE(
          "Failed to call axclrtEngineCreateContext(). Return code is: %d",
          static_cast<int32_t>(ret));
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void InitInput() {
    uint32_t group = 0;

    int32_t num_inputs = axclrtEngineGetNumInputs(io_info_);

    input_tensor_names_.resize(num_inputs);
    input_tensor_shapes_.reserve(num_inputs);
    input_tensor_.reserve(num_inputs);

    for (int32_t i = 0; i < num_inputs; ++i) {
      auto size_in_bytes = axclrtEngineGetInputSizeByIndex(io_info_, group, i);
      input_tensors_.emplace_back({size_in_bytes, AXCL_MEM_MALLOC_HUGE_FIRST});

      axclrtEngineIODims dims;
      auto ret = axclrtEngineGetInputDims(io_info_, group, i, &dims);
      if (ret != 0) {
        SHERPA_ONNX_LOGE(
            "Failed to call axclrtEngineGetInputDims(). Return code is: %d",
            static_cast<int32_t>(ret));
        SHERPA_ONNX_EXIT(-1);
      }

      input_tensors_shapes_.emplace_back(
          {dims.dims, dims.dims + dims.dimCount});

      input_tensor_names_[i] = axclrtEngineGetInputNameByIndex(io_info_, i);

      ret = axclrtEngineSetInputBufferByIndex(engine_io_guard_, i,
                                              input_tensors_[i], size_in_bytes);
      if (ret != 0) {
        SHERPA_ONNX_LOGE(
            "Failed to call axclrtEngineSetInputBufferByIndex(). Return code "
            "is: %d",
            static_cast<int32_t>(ret));
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  void InitOutput() {
    uint32_t group = 0;

    int32_t num_outputs = axclrtEngineGetNumOutputs(io_info_);

    output_tensor_names_.resize(num_outputs);
    output_tensor_shapes_.reserve(num_outputs);
    output_tensor_.reserve(num_outputs);

    for (int32_t i = 0; i < num_outputs; ++i) {
      auto size_in_bytes = axclrtEngineGetOutputSizeByIndex(io_info_, group, i);
      output_tensors_.emplace_back({size_in_bytes, AXCL_MEM_MALLOC_HUGE_FIRST});

      axclrtEngineIODims dims;
      ret = axclrtEngineGetOutputDims(io_info_, group, i, &dims);
      if (ret != 0) {
        SHERPA_ONNX_LOGE(
            "Failed to call axclrtEngineGetOutputDims(). Return code is: %d",
            static_cast<int32_t>(ret));
        SHERPA_ONNX_EXIT(-1);
      }

      output_tensors_shapes_.emplace_back(
          {dims.dims, dims.dims + dims.dimCount});
      output_tensor_names_[i] = axclrtEngineGetOutputNameByIndex(io_info_, i);

      ret = axclrtEngineSetOutputBufferByIndex(
          engine_io_guard_, i, output_tensors_[i], size_in_bytes);
      if (ret != 0) {
        SHERPA_ONNX_LOGE(
            "Failed to call axclrtEngineSetOutputBufferByIndex(). Return code "
            "is: %d",
            static_cast<int32_t>(ret));
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

 private:
  AxclManager manager_;
  std::unique_ptr<AxclEngineGuard> engine_guard_;
  std::unique_ptr<AxclEngineIOGuard> engine_io_guard_;

  bool model_loaded_ = false;
  uint64_t model_id_ = 0;
  uint64_t context_id_ = 0;

  axclrtEngineIOInfo io_info_ = nullptr;
  axclrtEngineIO io_ = nullptr;

  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> output_tensor_names_;

  std::vector<AxclDevicePtr> input_tensors_;
  std::vector<AxclDevicePtr> output_tensors_;

  std::vector<std::vector<int32_t>> input_tensor_shapes_;
  std::vector<std::vector<int32_t>> output_tensor_shapes_;
};

AxclModel::AxclModel(const std::string &filename, int32_t device_id /*= 0*/)
    : impl_(std::make_unique<Impl>(filename, device_id)) {}

AxclModel::AxclModel(const void *cpu_buf, size_t buf_len_in_bytes,
                     int32_t device_id /*= 0*/)
    : impl_(std::make_unique<Impl>(cpu_buf, buf_len_in_bytes, device_id)) {}

const std::vector<std::string> &AxclModel::InputTensorNames() const {
  return impl_->InputTensorNames();
}
const std::vector<std::string> &AxclModel::OutputTensorNames() const {
  return impl_->OutputTensorNames();
}

const std::vector<int32_t> &AxclModel::TensorShape(
    const std::string &name) const {
  return impl_->TensorShape(name);
}

int32_t AxclModel::TensorSizeInBytes(const std::string &name) const {
  return impl_->TensorSizeInBytes(name);
}

bool AxclModel::HasTensor(const std::string &name) const {
  return impl_->HasTensor(name);
}

bool AxclModel::SetInputTensorData(const std::string &name, const float *p,
                                   int32_t n) const {
  return impl_->SetInputTensorData(name, p, n);
}

bool AxclModel::SetInputTensorData(const std::string &name, const int32_t *p,
                                   int32_t n) const {
  return impl_->SetInputTensorData(name, p, n);
}

std::vector<float> AxclModel::GetOutputTensorData(
    const std::string &name) const {
  return impl_->GetOutputTensorData(name);
}

bool AxclModel::Run() const { return impl_->Run(); }

bool AxclModel::IsInitialized() const { return impl_->IsInitialized(); }

}  // namespace sherpa_onnx
