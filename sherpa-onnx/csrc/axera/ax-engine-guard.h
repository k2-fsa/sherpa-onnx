// sherpa-onnx/csrc/axera/ax-engine-guard.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXERA_AX_ENGINE_GUARD_H_
#define SHERPA_ONNX_CSRC_AXERA_AX_ENGINE_GUARD_H_
#include <cstdint>

namespace sherpa_onnx {

class AxEngineGuard {
 public:
  AxEngineGuard();
  ~AxEngineGuard();

  AxEngineGuard(const AxEngineGuard &) = delete;
  AxEngineGuard &operator=(const AxEngineGuard &) = delete;

  AxEngineGuard(AxEngineGuard &&) = delete;
  AxEngineGuard &operator=(AxEngineGuard &&) = delete;

 private:
  static thread_local int32_t count_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_AX_ENGINE_GUARD_H_
