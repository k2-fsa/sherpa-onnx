// sherpa-onnx/csrc/online-rnn-lm.h
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RNN_LM_H_
#define SHERPA_ONNX_CSRC_ONLINE_RNN_LM_H_

#include <memory>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-lm-config.h"
#include "sherpa-onnx/csrc/online-lm.h"

namespace sherpa_onnx {

class OnlineRnnLM : public OnlineLM {
 public:
  ~OnlineRnnLM() override;

  explicit OnlineRnnLM(const OnlineLMConfig &config);

  std::vector<Ort::Value> GetInitStates() override;

  /** ScoreToken a batch of sentences.
   *
   * @param x A 2-D tensor of shape (N, L) with data type int64.
   * @param lens A 2-D tensor of shape (N, L) with data type int64.
   * @param states It contains the states for the LM model
   * @return Return a pair containingo
   *          - log_softmax of NN LM
   *          - updated states
   *
   */
  std::pair<Ort::Value, std::vector<Ort::Value>> ScoreToken(
      Ort::Value x, std::vector<Ort::Value> states) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RNN_LM_H_
