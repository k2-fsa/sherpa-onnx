// sherpa-onnx/csrc/axcl/axcl-engine-io-guard.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axcl/axcl-engine-io-guard.h"

#include <cstdint>

#include "axcl.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

AxclEngineIOGuard::AxclEngineIOGuard(axclrtEngineIOInfo io_info) {
  axclError ret = axclrtEngineCreateIO(io_info, &io_);
  if (ret != 0) {
    SHERPA_ONNX_LOGE(
        "Failed to call axclrtEngineCreateIO(). Return code is: %d",
        static_cast<int32_t>(ret));
    SHERPA_ONNX_EXIT(-1);
  }

  initialized_ = true;
}

AxclEngineIOGuard::~AxclEngineIOGuard() {
  if (initialized_) {
    auto ret = axclrtEngineDestroyIO(io_);

    if (ret != 0) {
      SHERPA_ONNX_LOGE(
          "Failed to call axclrtEngineDestroyIO(). Return code is: %d",
          static_cast<int32_t>(ret));
      SHERPA_ONNX_EXIT(-1);
    }
  }
}

}  // namespace sherpa_onnx
