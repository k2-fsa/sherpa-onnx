// sherpa-onnx/csrc/axcl/offline-sense-voice-model-axcl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_OFFLINE_SENSE_VOICE_MODEL_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_OFFLINE_SENSE_VOICE_MODEL_AXCL_H_

#include <memory>
#include <vector>

#include "axcl.h"
#include "sherpa-onnx/csrc/axcl/ax_model_runner_axcl.hpp"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-sense-voice-model-meta-data.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelAxcl {
 public:
  ~OfflineSenseVoiceModelAxcl();

  explicit OfflineSenseVoiceModelAxcl(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineSenseVoiceModelAxcl(Manager *mgr, const OfflineModelConfig &config);

  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) const;

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_OFFLINE_SENSE_VOICE_MODEL_AXCL_H_