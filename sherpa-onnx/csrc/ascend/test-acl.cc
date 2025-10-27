// sherpa-onnx/csrc/ascend/test-acl.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-onnx/csrc/ascend/macros.h"
#include "sherpa-onnx/csrc/ascend/offline-sense-voice-model-ascend.h"
#include "sherpa-onnx/csrc/ascend/utils.h"

namespace sherpa_onnx {

static void TestAcl() {
  Acl acl;

  int32_t device_id = 0;
  aclError ret = aclrtSetDevice(device_id);
  SHERPA_ONNX_ASCEND_CHECK(
      ret, "Failed to call aclrtSetDevice with device id: %d", device_id);

  AclContext context(device_id);

  ret = aclrtSetCurrentContext(context.Get());
  SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");

  OfflineSenseVoiceModelAscend model;
}

}  // namespace sherpa_onnx

int main() {
  sherpa_onnx::TestAcl();
  return 0;
}
