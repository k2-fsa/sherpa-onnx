// sherpa-onnx/csrc/axera/offline-sense-voice-model-axera.h
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_SENSE_VOICE_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_SENSE_VOICE_MODEL_AXERA_H_

#include <memory>
#include <vector>

#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-sense-voice-model-meta-data.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelAxera {
 public:
  ~OfflineSenseVoiceModelAxera();

  explicit OfflineSenseVoiceModelAxera(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineSenseVoiceModelAxera(Manager *mgr, const OfflineModelConfig &config);

  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) const;

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_SENSE_VOICE_MODEL_AXERA_H_