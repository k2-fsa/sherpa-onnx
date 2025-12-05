// sherpa-onnx/csrc/axcl/axcl-engine-guard.h
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axcl/axcl-engine-guard.h"

#include <cstdint>

#include "axcl.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {
AxclEngineGuard::AxclEngineGuard(axclrtEngineVNpuKind npuKind) {
  axclError ret = axclrtEngineInit(npuKind);
  if (ret != 0) {
    SHERPA_ONNX_LOGE("Failed to call axclrtEngineInit(). Return code is: %d",
                     static_cast<int32_t>(ret));
    SHERPA_ONNX_EXIT(-1);
  }

  initialized_ = true;
}

AxclEngineGuard::~AxclEngineGuard() {
  if (initialized_) {
    auto ret = axclrtEngineFinalize();

    if (ret != 0) {
      SHERPA_ONNX_LOGE(
          "Failed to call axclrtEngineFinalize(). Return code is: %d",
          static_cast<int32_t>(ret));
      SHERPA_ONNX_EXIT(-1);
    }
  }
}
}  // namespace sherpa_onnx
