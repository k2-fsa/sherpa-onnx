// sherpa/csrc/online-lstm-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_LSTM_TRANSDUCER_H_
#define SHERPA_ONNX_CSRC_ONLINE_LSTM_TRANSDUCER_H_

#include "online-transducer-model-config.h"
#include "online-transducer-model.h"
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

class OnlineLstmTransducerModel : OnlineTransducerModel {
 public:
  explicit OnlineLstmTransducerModel(
      const OnlineTransducerModelConfig &model_config);

  Ort::Value StackStates(const std::vector<Ort::Value> &states) const override;

  std::vector<Ort::Value> UnStackStates(Ort::Value states) const override;

  Ort::Value GetEncoderInitStates() override;

  std::pair<Ort::Value, std::vector<Ort::Value>> RunEncoder(
      const Ort::Value &features,
      const std::vector<Ort::Value> &states) override;

  Ort::Value RunDecoder(const Ort::Value &decoder_input) override;

  Ort::Value RunJoiner(const Ort::Value &encoder_out,
                       const Ort::Value &decoder_out) override;

  int32_t ContextSize() const override;

  int32_t ChunkSize() const override;

  int32_t ChunkShift() const override;

  int32_t VocabSize() const override;

 private:
  void InitEncoder(const std::string &encoder_filename);
  void InitDecoder(const std::string &decoder_filename);
  void InitJoiner(const std::string &joiner_filename);

 private:
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;
  std::unique_ptr<Ort::Session> joiner_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<std::string> joiner_input_names_;
  std::vector<const char *> joiner_input_names_ptr_;

  std::vector<std::string> joiner_output_names_;
  std::vector<const char *> joiner_output_names_ptr_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_LSTM_TRANSDUCER_H_
