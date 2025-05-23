// sherpa-onnx/csrc/offline-source-separation-spleeter-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_IMPL_H_

namespace sherpa_onnx {

class OfflineSourceSeparationSpleeterImpl : public OfflineSourceSeparationImpl {
 public:
  OfflineSourceSeparationSpleeterImpl(
      const OfflineSourceSeparationConfig &config) {}

  template <typename Manager>
  OfflineSourceSeparationSpleeterImpl(
      Manager *mgr, const OfflineSourceSeparationConfig &config) {}

  OfflineSourceSeparationOutput Process(
      const OfflineSourceSeparationInput &input) const override {
    return {};
  }

  int32_t GetOutputSampleRate() const override { return 44100; }

  int32_t GetNumberOfStems() const override { return 2; }

 private:
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_IMPL_H_
