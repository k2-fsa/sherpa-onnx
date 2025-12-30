// sherpa-onnx/csrc/axcl/axcl-engine-io-info-guard.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_AXCL_ENGINE_IO_INFO_GUARD_H_
#define SHERPA_ONNX_CSRC_AXCL_AXCL_ENGINE_IO_INFO_GUARD_H_
#include <cstdint>

#include "axcl.h"  // NOLINT

namespace sherpa_onnx {

class AxclEngineIOInfoGuard {
 public:
  explicit AxclEngineIOInfoGuard(uint64_t model_id);
  ~AxclEngineIOInfoGuard();

  AxclEngineIOInfoGuard(const AxclEngineIOInfoGuard &) = delete;
  AxclEngineIOInfoGuard &operator=(const AxclEngineIOInfoGuard &) = delete;
  AxclEngineIOInfoGuard(AxclEngineIOInfoGuard &&) = delete;
  AxclEngineIOInfoGuard &operator=(AxclEngineIOInfoGuard &&) = delete;

  operator axclrtEngineIOInfo() { return io_info_; }

 private:
  bool initialized_ = false;
  axclrtEngineIOInfo io_info_ = nullptr;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_AXCL_ENGINE_IO_INFO_GUARD_H_
