// sherpa-onnx/csrc/qnn/qnn-backend.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_QNN_QNN_BACKEND_H_
#define SHERPA_ONNX_CSRC_QNN_QNN_BACKEND_H_

#include <memory>
#include <string>

#include "QnnInterface.h"

namespace sherpa_onnx {

class QnnBackend {
 public:
  explicit QnnBackend(const std::string &backend_lib, bool debug);
  ~QnnBackend();

  void InitContext() const;
  void InitContext(Qnn_ContextHandle_t context_handle) const;
  Qnn_LogHandle_t LogHandle() const;
  Qnn_BackendHandle_t BackendHandle() const;
  Qnn_DeviceHandle_t DeviceHandle() const;
  Qnn_ContextHandle_t ContextHandle() const;
  QNN_INTERFACE_VER_TYPE QnnInterface() const;
  QnnLog_Level_t LogLevel() const;
  bool IsInitialized() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_QNN_BACKEND_H_
