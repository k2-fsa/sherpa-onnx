// sherpa-onnx/csrc/axcl/offline-tts-supertonic-model-axcl.h
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#ifndef SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_SUPERTONIC_MODEL_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_SUPERTONIC_MODEL_AXCL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-supertonic-model.h"

namespace sherpa_onnx {

class OfflineTtsSupertonicModelAxcl {
 public:
  ~OfflineTtsSupertonicModelAxcl();

  explicit OfflineTtsSupertonicModelAxcl(
      const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsSupertonicModelAxcl(Manager *mgr,
                                 const OfflineTtsModelConfig &config);

  const SupertonicConfig &GetConfig() const;
  int32_t GetSampleRate() const;

  // Fixed-shape run methods. All input/output tensors are flat vectors.
  // Shape is implied by the fixed model metadata.

  // Inputs: text_ids [1, 320], style_dp [1, 8, 16], text_mask [1, 1, 320]
  // Output: duration [1]
  std::vector<float> RunDurationPredictor(
      const std::vector<int64_t> &text_ids, const std::vector<float> &style_dp,
      const std::vector<float> &text_mask) const;

  // Inputs: text_ids [1, 320], style_ttl [1, 50, 256], text_mask [1, 1, 320]
  // Output: text_emb [1, 256, 320]
  std::vector<float> RunTextEncoder(const std::vector<int64_t> &text_ids,
                                    const std::vector<float> &style_ttl,
                                    const std::vector<float> &text_mask) const;

  // Inputs: noisy_latent [1, latent_dim, 300],
  //         text_emb [1, 256, 320],
  //         style_ttl [1, 50, 256],
  //         latent_mask [1, 1, 300],
  //         text_mask [1, 1, 320],
  //         current_step [1],
  //         total_step [1]
  // Output: denoised_latent [1, latent_dim, 300]
  std::vector<float> RunVectorEstimator(
      const std::vector<float> &noisy_latent,
      const std::vector<float> &current_step,
      const std::vector<float> &text_emb,
      const std::vector<float> &style_ttl,
      const std::vector<float> &latent_mask,
      const std::vector<float> &text_mask,
      const std::vector<float> &total_step) const;

  // Input: latent [1, latent_dim, 300]
  // Output: wav [1, max_samples] (fixed shape, needs truncation)
  std::vector<float> RunVocoder(const std::vector<float> &latent) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_SUPERTONIC_MODEL_AXCL_H_
