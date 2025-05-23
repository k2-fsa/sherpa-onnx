// sherpa-onnx/csrc/offline-source-separation-spleeter-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_IMPL_H_

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-source-separation.h"

namespace sherpa_onnx {

class OfflineSourceSeparationSpleeterImpl : public OfflineSourceSeparationImpl {
 public:
  OfflineSourceSeparationSpleeterImpl(
      const OfflineSourceSeparationConfig &config) {
    SHERPA_ONNX_LOGE("created!");
  }

  template <typename Manager>
  OfflineSourceSeparationSpleeterImpl(
      Manager *mgr, const OfflineSourceSeparationConfig &config) {}

  OfflineSourceSeparationOutput Process(
      const OfflineSourceSeparationInput &input) const override {
    SHERPA_ONNX_LOGE("processing!");
    return {};
  }

  int32_t GetOutputSampleRate() const override { return 44100; }

  int32_t GetNumberOfStems() const override { return 2; }

 private:
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_IMPL_H_
