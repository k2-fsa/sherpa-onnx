// sherpa-onnx/csrc/axcl/axcl-engine-guard.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_AXCL_ENGINE_GUARD_H_
#define SHERPA_ONNX_CSRC_AXCL_AXCL_ENGINE_GUARD_H_
#include "axcl.h"  // NOLINT

namespace sherpa_onnx {

class AxclEngineGuard {
 public:
  explicit AxclEngineGuard(axclrtEngineVNpuKind npuKind);
  ~AxclEngineGuard();

  AxclEngineGuard(const AxclEngineGuard &) = delete;
  AxclEngineGuard &operator=(const AxclEngineGuard &) = delete;
  AxclEngineGuard(AxclEngineGuard &&) = delete;
  AxclEngineGuard &operator=(AxclEngineGuard &&) = delete;

 private:
  bool initialized_ = false;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_AXCL_ENGINE_GUARD_H_
