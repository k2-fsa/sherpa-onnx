// sherpa-onnx/csrc/qnn/offline-parakeet-tdt-model-qnn.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_QNN_OFFLINE_PARAKEET_TDT_MODEL_QNN_H_
#define SHERPA_ONNX_CSRC_QNN_OFFLINE_PARAKEET_TDT_MODEL_QNN_H_

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineParakeetTdtModelQnn {
 public:
  ~OfflineParakeetTdtModelQnn();

  explicit OfflineParakeetTdtModelQnn(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineParakeetTdtModelQnn(Manager *mgr, const OfflineModelConfig &config);

  /** Run the encoder.
   *
   * @param features A flat vector of shape (num_frames, feat_dim),
   *                 already mean-variance normalized with per_feature.
   *
   * @return A flat vector of shape (num_encoder_frames, encoder_out_dim).
   */
  std::vector<float> RunEncoder(std::vector<float> features) const;

  /** Get the initial decoder states (all zeros).
   *
   * @return A vector of state tensors. For LSTM, it contains [h, c].
   */
  std::vector<std::vector<float>> GetDecoderInitStates() const;

  /** Run the decoder with states.
   *
   * @param token  A single token id.
   * @param states  The decoder states (e.g. [h, c] for LSTM).
   *
   * @return A pair of (decoder_out, next_states).
   */
  std::pair<std::vector<float>, std::vector<std::vector<float>>> RunDecoder(
      int32_t token, std::vector<std::vector<float>> states) const;

  /** Run the joiner.
   *
   * @param encoder_out  Pointer to a single encoder output frame.
   * @param decoder_out  The decoder output vector.
   *
   * @return A flat vector of shape (vocab_size + num_durations,).
   */
  std::vector<float> RunJoiner(const float *encoder_out,
                               const std::vector<float> &decoder_out) const;

  int32_t SubsamplingFactor() const;
  int32_t EncoderDim() const;
  int32_t FeatDim() const;
  int32_t NumEncoderFrames(int32_t num_frames) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_OFFLINE_PARAKEET_TDT_MODEL_QNN_H_
