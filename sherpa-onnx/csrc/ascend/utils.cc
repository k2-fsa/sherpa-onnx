// sherpa-onnx/csrc/ascend/utils.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/ascend/utils.h"

#include "sherpa-onnx/csrc/ascend/macros.h"

namespace sherpa_onnx {

Acl::Acl() {
  aclError ret = aclInit(nullptr);
  SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclInit");
  initialized_ = true;
}

Acl::~Acl() {
  if (initialized_) {
    aclError ret = aclFinalize();
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclFinalize");
  }
}

AclContext::AclContext(int32_t device_id) {
  aclError ret = aclrtCreateContext(&context_, device_id);
  SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtCreateContext");
}

AclContext::~AclContext() {
  if (context_) {
    aclError ret = aclrtDestroyContext(context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtDestroyContext");
  }
}

aclrtContext AclContext::Get() const { return context_; }

AclDevicePtr::AclDevicePtr(
    size_t size, aclrtMemMallocPolicy policy /*= ACL_MEM_MALLOC_HUGE_FIRST*/) {
  aclError ret = aclrtMalloc(&p_, size, policy);

  SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMalloc with size: %zu",
                           size);
}

AclDevicePtr::~AclDevicePtr() {
  if (p_) {
    aclError ret = aclrtFree(p_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtFree");
  }
}

AclHostPtr::AclHostPtr(size_t size) {
  aclError ret = aclrtMallocHost(&p_, size);

  SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMallocHost with size: %zu",
                           size);
}

AclHostPtr::~AclHostPtr() {
  if (p_) {
    aclError ret = aclrtFreeHost(p_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtFreeHost");
  }
}

}  // namespace sherpa_onnx
