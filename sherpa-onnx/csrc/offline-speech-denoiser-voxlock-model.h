// sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model.h
//
// Copyright (c)  2026  Arbaz Siddiqui
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_MODEL_H_

#include <memory>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-speech-denoiser-model-config.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model-meta-data.h"

namespace sherpa_onnx {

class OfflineSpeechDenoiserVoxLockModel {
 public:
  ~OfflineSpeechDenoiserVoxLockModel();
  explicit OfflineSpeechDenoiserVoxLockModel(
      const OfflineSpeechDenoiserModelConfig &config);

  template <typename Manager>
  OfflineSpeechDenoiserVoxLockModel(
      Manager *mgr, const OfflineSpeechDenoiserModelConfig &config);

  using States = std::vector<Ort::Value>;

  /* Run the enroll model on a voice sample of the caller.
   *
   * @param samples Voice sample at GetMetaData().sample_rate. It must have at
   *                least GetMetaData().min_voice_sample_length samples.
   * @param n Number of samples.
   *
   * @return Return a pair containing:
   *  - enroll_mag, a tensor of shape (1, num_freq_bins, num_frames)
   *  - context, a tensor of shape (1, context_dim, num_context_frames)
   */
  std::pair<Ort::Value, Ort::Value> Enroll(const float *samples,
                                           int32_t n) const;

  // Zero LSTM states h and c
  States GetInitStates() const;

  /* Run the chunk model.
   *
   * @param wav A tensor of shape (1, (K - 1) * hop_length + n_fft)
   * @param enroll_mag Returned by Enroll()
   * @param context Returned by Enroll()
   * @param states LSTM states from GetInitStates() or a previous call
   *
   * @return Return a pair containing:
   *  - frames, a tensor of shape (1, K, n_fft). Each frame is windowed by
   *    the synthesis window and is ready for overlap-add.
   *  - next states
   */
  std::pair<Ort::Value, States> Run(Ort::Value wav, Ort::Value enroll_mag,
                                    Ort::Value context, States states) const;

  const OfflineSpeechDenoiserVoxLockModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_MODEL_H_
