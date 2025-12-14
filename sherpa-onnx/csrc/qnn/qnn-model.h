// sherpa-onnx/csrc/qnn/qnn-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_QNN_QNN_MODEL_H_
#define SHERPA_ONNX_CSRC_QNN_QNN_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "QnnInterface.h"

namespace sherpa_onnx {

class QnnBackend;

struct BinaryContextTag {};

class QnnModel {
 public:
  QnnModel(const std::string &model_so, const QnnBackend *backend, bool debug);
  QnnModel(const std::string &binary_context_file,
           const std::string &system_lib, const QnnBackend *backend,
           BinaryContextTag tag, bool debug);
  ~QnnModel();

  bool SaveBinaryContext(const std::string &filename) const;

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

#endif  // SHERPA_ONNX_CSRC_QNN_QNN_MODEL_H_
