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
  ~AxclModel();

  const std::vector<std::string> &InputTensorNames() const;
  const std::vector<std::string> &OutputTensorNames() const;

  std::vector<int32_t> TensorShape(const std::string &name) const;
  int32_t TensorSizeInBytes(const std::string &name) const;
  int32_t NumInputs() const;
  int32_t NumOutputs() const;
  const std::string &InputName(int32_t i) const;
  const std::string &OutputName(int32_t i) const;
  int32_t InputSizeInBytes(int32_t i) const;
  int32_t OutputSizeInBytes(int32_t i) const;

  bool HasTensor(const std::string &name) const;

  bool SetInputTensorData(const std::string &name, const float *p,
                          int32_t n) const;

  bool SetInputTensorData(const std::string &name, const int32_t *p,
                          int32_t n) const;

  bool SetInputTensorDataRaw(const std::string &name, const void *p,
                             int32_t nbytes) const;

  std::vector<float> GetOutputTensorData(const std::string &name) const;

  std::vector<uint8_t> GetOutputTensorDataRaw(const std::string &name) const;

  bool Run() const;
  bool IsInitialized() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_AXCL_MODEL_H_
