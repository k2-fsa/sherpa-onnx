// sherpa-onnx/csrc/qnn/online-nemo-transducer-model-qnn.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_ONLINE_NEMO_TRANSDUCER_MODEL_QNN_H_
#define SHERPA_ONNX_CSRC_QNN_ONLINE_NEMO_TRANSDUCER_MODEL_QNN_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-stream-state.h"

namespace sherpa_onnx {

class OnlineNemoTransducerModelQnn {
 public:
  ~OnlineNemoTransducerModelQnn();

  explicit OnlineNemoTransducerModelQnn(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineNemoTransducerModelQnn(Manager *mgr, const OnlineModelConfig &config);

  std::vector<OnlineStreamStateTensor> GetEncoderInitStates() const;

  std::vector<float> RunEncoder(std::vector<float> features,
                                int32_t num_frames,
                                std::vector<OnlineStreamStateTensor> *states,
                                int32_t prompt_index = -1) const;

  // Run the LSTM decoder.
  //
  // @param token  A single token id.
  // @param states  Decoder states from previous call or GetDecoderInitState().
  //
  // @return (decoder_out, next_states)
  std::pair<std::vector<float>, std::vector<std::vector<float>>> RunDecoder(
      int32_t token, std::vector<std::vector<float>> states) const;

  std::vector<float> RunJoiner(const float *encoder_out,
                               const std::vector<float> &decoder_out) const;

  int32_t WindowSize() const;
  int32_t WindowShift() const;
  int32_t VocabSize() const;
  int32_t FeatureDim() const;
  int32_t EncoderDim() const;
  int32_t DecoderDim() const;
  int32_t SubsamplingFactor() const;
  const std::string &NormalizationType() const;

  // Get the prompt ID for a given language (e.g., "en-US", "zh-CN").
  // Returns the prompt ID for "auto" for empty or unknown language
  // (logs warning for unknown).
  int32_t GetLanguagePromptId(const std::string &language) const;

  // Get the initial zero-filled decoder states (e.g., [h, c] for LSTM).
  std::vector<std::vector<float>> GetDecoderInitState() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_ONLINE_NEMO_TRANSDUCER_MODEL_QNN_H_
