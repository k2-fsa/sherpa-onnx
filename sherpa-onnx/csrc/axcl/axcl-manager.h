// sherpa-onnx/csrc/axcl/axcl-manager.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_AXCL_MANAGER_H_
#define SHERPA_ONNX_CSRC_AXCL_AXCL_MANAGER_H_

#include <atomic>
#include <mutex>

namespace sherpa_onnx {

class AxclManager {
 public:
  explicit AxclManager(const char *config = nullptr);
  ~AxclManager();

  AxclManager(const AxclManager &) = delete;
  AxclManager &operator=(const AxclManager &) = delete;

  AxclManager(AxclManager &&) = delete;
  AxclManager &operator=(AxclManager &&) = delete;

 private:
  static std::mutex mutex_;
  static std::atomic<int> instanceCount_;
};
}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_AXCL_AXCL_MANAGER_H_
