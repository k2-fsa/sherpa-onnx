// sherpa-onnx/csrc/axcl/offline-tts-kokoro-model-axcl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_KOKORO_MODEL_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_KOKORO_MODEL_AXCL_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts-kokoro-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

class OfflineTtsKokoroModelAxcl {
 public:
  ~OfflineTtsKokoroModelAxcl();

  explicit OfflineTtsKokoroModelAxcl(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsKokoroModelAxcl(Manager *mgr, const OfflineTtsModelConfig &config);

  std::vector<float> Run(const std::vector<int64_t>& x, int64_t sid = 0, float speed = 1.0) const;

  const OfflineTtsKokoroModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_KOKORO_MODEL_AXCL_H_

