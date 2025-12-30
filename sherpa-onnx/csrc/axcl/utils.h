// sherpa-onnx/csrc/axcl/utils.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_UTILS_H_
#define SHERPA_ONNX_CSRC_AXCL_UTILS_H_

#include "axcl.h"  // NOLINT

namespace sherpa_onnx {

class AxclDevicePtr {
 public:
  explicit AxclDevicePtr(
      size_t size, axclrtMemMallocPolicy policy = AXCL_MEM_MALLOC_HUGE_FIRST);

  ~AxclDevicePtr();

  AxclDevicePtr(const AxclDevicePtr &) = delete;
  AxclDevicePtr &operator=(const AxclDevicePtr &) = delete;

  AxclDevicePtr(AxclDevicePtr &&other) {
    p_ = other.p_;
    size_ = other.size_;

    other.p_ = nullptr;
    other.size_ = 0;
  }
  AxclDevicePtr &operator=(AxclDevicePtr &&other) {
    if (this == &other) {
      return *this;
    }
    Release();
    p_ = other.p_;
    size_ = other.size_;

    other.p_ = nullptr;
    other.size_ = 0;

    return *this;
  }

  void Release();

  void *Get() const { return p_; }
  operator void *() { return p_; }

  size_t Size() const { return size_; }

 private:
  void *p_ = nullptr;
  size_t size_ = 0;  // in bytes
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_UTILS_H_
