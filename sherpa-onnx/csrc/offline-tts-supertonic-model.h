// sherpa-onnx/csrc/offline-tts-supertonic-model.h
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

struct SupertonicConfig {
  struct AEConfig {
    int32_t sample_rate;
    int32_t base_chunk_size;
  } ae;

  struct TTLConfig {
    int32_t chunk_compress_factor;
    int32_t latent_dim;
  } ttl;
};

struct SupertonicStyle {
  std::vector<float> ttl_data;
  std::vector<float> dp_data;
  std::vector<int64_t> ttl_shape;
  std::vector<int64_t> dp_shape;
};

class OfflineTtsSupertonicModel {
 public:
  ~OfflineTtsSupertonicModel();

  explicit OfflineTtsSupertonicModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsSupertonicModel(Manager *mgr, const OfflineTtsModelConfig &config);

  const SupertonicConfig &GetConfig() const;
  int32_t GetSampleRate() const;

  // Get ONNX sessions for inference
  Ort::Session *GetDurationPredictorSession() const;
  Ort::Session *GetTextEncoderSession() const;
  Ort::Session *GetVectorEstimatorSession() const;
  Ort::Session *GetVocoderSession() const;

  // GPU IO Binding support
  bool UseCudaIOBinding() const;
  const Ort::MemoryInfo &GetCpuMemoryInfo() const;
  const Ort::MemoryInfo *GetCudaMemoryInfo() const;
  std::string GetProvider() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_H_
