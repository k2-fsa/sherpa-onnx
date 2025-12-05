// sherpa-onnx/csrc/axcl/utils.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axcl/utils.h"

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

AxclDevicePtr::~AxclDevicePtr() {
  auto ret = axclrtFree(p_);
  if (ret != 0) {
    SHERPA_ONNX_LOGE("Failed to call axclrtFree(). Return code: %d",
                     static_cast<int32_t>(ret));
    SHERPA_ONNX_EXIT(-1);
  }
}

}  // namespace sherpa_onnx
