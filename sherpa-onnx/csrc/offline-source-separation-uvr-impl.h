// sherpa-onnx/csrc/offline-source-separation-uvr-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_IMPL_H_

#include "Eigen/Dense"
#include "kaldi-native-fbank/csrc/istft.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-source-separation-uvr-model.h"
#include "sherpa-onnx/csrc/offline-source-separation.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

class OfflineSourceSeparationUvrImpl : public OfflineSourceSeparationImpl {
 public:
  OfflineSourceSeparationUvrImpl(const OfflineSourceSeparationConfig &config)
      : config_(config), model_(config_.model) {}

  template <typename Manager>
  OfflineSourceSeparationUvrImpl(Manager *mgr,
                                 const OfflineSourceSeparationConfig &config)
      : config_(config), model_(mgr, config_.model) {}

  OfflineSourceSeparationOutput Process(
      const OfflineSourceSeparationInput &input) const override {
    return {};
  }

  int32_t GetOutputSampleRate() const override {
    return model_.GetMetaData().sample_rate;
  }

  int32_t GetNumberOfStems() const override {
    return model_.GetMetaData().num_stems;
  }

 private:
  OfflineSourceSeparationConfig config_;
  OfflineSourceSeparationUvrModel model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_IMPL_H_
