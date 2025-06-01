// sherpa-onnx/csrc/offline-source-separation.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-source-separation-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineSourceSeparationConfig {
  OfflineSourceSeparationModelConfig model;

  OfflineSourceSeparationConfig() = default;

  explicit OfflineSourceSeparationConfig(
      const OfflineSourceSeparationModelConfig &model)
      : model(model) {}

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

struct MultiChannelSamples {
  // data[i] is for the i-th channel
  //
  // each sample is in the range [-1, 1]
  std::vector<std::vector<float>> data;
};

struct OfflineSourceSeparationInput {
  MultiChannelSamples samples;

  int32_t sample_rate;
};

struct OfflineSourceSeparationOutput {
  std::vector<MultiChannelSamples> stems;

  int32_t sample_rate;
};

class OfflineSourceSeparationImpl;

class OfflineSourceSeparation {
 public:
  ~OfflineSourceSeparation();

  explicit OfflineSourceSeparation(const OfflineSourceSeparationConfig &config);

  template <typename Manager>
  OfflineSourceSeparation(Manager *mgr,
                          const OfflineSourceSeparationConfig &config);

  OfflineSourceSeparationOutput Process(
      const OfflineSourceSeparationInput &input) const;

  int32_t GetOutputSampleRate() const;

  // e.g., it is 2 for 2stems from spleeter
  int32_t GetNumberOfStems() const;

 private:
  std::unique_ptr<OfflineSourceSeparationImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_H_
