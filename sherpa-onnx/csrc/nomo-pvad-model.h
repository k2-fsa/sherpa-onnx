// sherpa-onnx/csrc/nomo-pvad-model.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_NOMO_PVAD_MODEL_H_
#define SHERPA_ONNX_CSRC_NOMO_PVAD_MODEL_H_

#include <memory>

#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {

/** Target-speaker VAD backed by https://huggingface.co/TuyaNomo/nomo-pvad
 *
 * The model is trained to answer a narrower question than Silero VAD: given
 * an enrollment embedding of a target speaker, is that speaker talking?
 *
 * Internally it keeps the last `window_samples` (2.4 s by default) of audio
 * and re-scores the whole window for every incoming `chunk_samples` (160 ms).
 * Only the probability of the newest chunk is used, matching the reference
 * implementation from the model authors.
 */
class NomoPvadModel : public VadModel {
 public:
  explicit NomoPvadModel(const VadModelConfig &config);

  template <typename Manager>
  NomoPvadModel(Manager *mgr, const VadModelConfig &config);

  ~NomoPvadModel() override;

  // Drop the audio history and the speech/silence state machine.
  void Reset() override;

  /**
   * @param samples Pointer to a 1-d array containing audio samples.
   *                Each sample should be normalized to the range [-1, 1].
   * @param n Number of samples.
   *
   * @return Return true if the target speaker is talking.
   */
  bool IsSpeech(const float *samples, int32_t n) override;

  // Return the probability that the target speaker is talking in the newest
  // chunk. `n` samples are appended to the internal audio history first.
  float Compute(const float *samples, int32_t n) override;

  // Both are 2560 for the released model, i.e., 160 ms at 16 kHz.
  int32_t WindowSize() const override;

  int32_t WindowShift() const override;

  int32_t MinSilenceDurationSamples() const override;
  int32_t MinSpeechDurationSamples() const override;

  void SetMinSilenceDuration(float s) override;
  void SetThreshold(float threshold) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_NOMO_PVAD_MODEL_H_
