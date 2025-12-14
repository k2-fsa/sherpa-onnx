// sherpa-onnx/csrc/axcl/utils.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axcl/utils.h"

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

AxclDevicePtr::AxclDevicePtr(
    size_t size,
    axclrtMemMallocPolicy policy /*= AXCL_MEM_MALLOC_HUGE_FIRST*/) {
  auto ret = axclrtMalloc(&p_, size, policy);
  if (ret != 0) {
    SHERPA_ONNX_LOGE("Failed to call axclrtMalloc(). Return code: %d",
                     static_cast<int32_t>(ret));
    SHERPA_ONNX_EXIT(-1);
  }

  size_ = size;
}

void AxclDevicePtr::Release() {
  if (!p_) {
    return;
  }

  auto ret = axclrtFree(p_);
  if (ret != 0) {
    SHERPA_ONNX_LOGE("Failed to call axclrtFree(). Return code: %d",
                     static_cast<int32_t>(ret));
    SHERPA_ONNX_EXIT(-1);
  }
  p_ = nullptr;
  size_ = 0;
}

AxclDevicePtr::~AxclDevicePtr() { Release(); }

}  // namespace sherpa_onnx
