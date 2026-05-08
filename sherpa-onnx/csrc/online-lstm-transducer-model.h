// sherpa-onnx/csrc/online-lstm-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_LSTM_TRANSDUCER_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_LSTM_TRANSDUCER_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

class OnlineLstmTransducerModel : public OnlineTransducerModel {
 public:
  explicit OnlineLstmTransducerModel(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineLstmTransducerModel(Manager *mgr, const OnlineModelConfig &config);

  std::vector<Ort::Value> StackStates(
      const std::vector<std::vector<Ort::Value>> &states) const override;

  std::vector<std::vector<Ort::Value>> UnStackStates(
      const std::vector<Ort::Value> &states) const override;

  std::vector<Ort::Value> GetEncoderInitStates() override;

  std::pair<Ort::Value, std::vector<Ort::Value>> RunEncoder(
      Ort::Value features, std::vector<Ort::Value> states,
      Ort::Value processed_frames) override;

  Ort::Value RunDecoder(Ort::Value decoder_input) override;

  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out) override;

  int32_t ContextSize() const override { return context_size_; }

  int32_t ChunkSize() const override { return T_; }

  int32_t ChunkShift() const override { return decode_chunk_len_; }

  int32_t VocabSize() const override { return vocab_size_; }
  OrtAllocator *Allocator() override { return allocator_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length);
  void InitDecoder(void *model_data, size_t model_data_length);
  void InitJoiner(void *model_data, size_t model_data_length);

 private:
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

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

  OnlineModelConfig config_;

  int32_t num_encoder_layers_ = 0;
  int32_t T_ = 0;
  int32_t decode_chunk_len_ = 0;
  int32_t rnn_hidden_size_ = 0;
  int32_t d_model_ = 0;
  int32_t context_size_ = 0;
  int32_t vocab_size_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_LSTM_TRANSDUCER_MODEL_H_
