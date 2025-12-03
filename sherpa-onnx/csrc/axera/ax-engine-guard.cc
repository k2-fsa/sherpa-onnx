// sherpa-onnx/csrc/axera/ax-engine-guard.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"

#include <cstring>

#include "ax_engine_api.h"  // NOLINT
#include "ax_sys_api.h"     // NOLINT
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

thread_local int32_t AxEngineGuard::count_ = 0;

AxEngineGuard::AxEngineGuard() {
  if (count_ == 0) {
    auto ret = AX_SYS_Init();
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to call AX_SYS_Init. ret code: %d",
                       static_cast<int32_t>(ret));

      SHERPA_ONNX_EXIT(-1);
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    ret = AX_ENGINE_Init(&npu_attr);

    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to call AX_ENGINE_Init. ret code: %d",
                       static_cast<int32_t>(ret));

      SHERPA_ONNX_EXIT(-1);
    }
  }

  ++count_;
}

AxEngineGuard::~AxEngineGuard() {
  --count_;
  if (count_ == 0) {
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
  }
}

}  // namespace sherpa_onnx
