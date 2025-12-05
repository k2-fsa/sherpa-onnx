// sherpa-onnx/csrc/axcl/axcl-model.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_AXCL_MODEL_H_
#define SHERPA_ONNX_CSRC_AXCL_AXCL_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

class AxclModel {
 public:
  explicit AxclModel(const std::string &filename, int32_t device_id = 0);

  AxclModel(const void *cpu_buf, size_t buf_len_in_bytes,
            int32_t device_id = 0);

  const std::vector<std::string> &InputTensorNames() const;
  const std::vector<std::string> &OutputTensorNames() const;

  std::vector<int32_t> TensorShape(const std::string &name) const;
  int32_t TensorSizeInBytes(const std::string &name) const;

  bool HasTensor(const std::string &name) const;

  bool SetInputTensorData(const std::string &name, const float *p,
                          int32_t n) const;

  bool SetInputTensorData(const std::string &name, const int32_t *p,
                          int32_t n) const;

  std::vector<float> GetOutputTensorData(const std::string &name) const;

  bool Run() const;
  bool IsInitialized() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_AXCL_MODEL_H_
