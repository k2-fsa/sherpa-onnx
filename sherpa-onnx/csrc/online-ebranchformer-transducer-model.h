// sherpa-onnx/csrc/online-ebranchformer-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
//                2025  Brno University of Technology (author: Karel Vesely)
#ifndef SHERPA_ONNX_CSRC_ONLINE_EBRANCHFORMER_TRANSDUCER_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_EBRANCHFORMER_TRANSDUCER_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

class OnlineEbranchformerTransducerModel : public OnlineTransducerModel {
 public:
  explicit OnlineEbranchformerTransducerModel(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineEbranchformerTransducerModel(Manager *mgr,
                                     const OnlineModelConfig &config);

  std::vector<Ort::Value> StackStates(
      const std::vector<std::vector<Ort::Value>> &states) const override;

  std::vector<std::vector<Ort::Value>> UnStackStates(
      const std::vector<Ort::Value> &states) const override;

  std::vector<Ort::Value> GetEncoderInitStates() override;

  void SetFeatureDim(int32_t feature_dim) override {
    feature_dim_ = feature_dim;
  }

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
  Ort::SessionOptions encoder_sess_opts_;
  Ort::SessionOptions decoder_sess_opts_;
  Ort::SessionOptions joiner_sess_opts_;

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

  int32_t decode_chunk_len_ = 0;
  int32_t T_ = 0;

  int32_t num_hidden_layers_ = 0;
  int32_t hidden_size_ = 0;
  int32_t intermediate_size_ = 0;
  int32_t csgu_kernel_size_ = 0;
  int32_t merge_conv_kernel_ = 0;
  int32_t left_context_len_ = 0;
  int32_t num_heads_ = 0;
  int32_t head_dim_ = 0;

  int32_t context_size_ = 0;
  int32_t vocab_size_ = 0;
  int32_t feature_dim_ = 80;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_EBRANCHFORMER_TRANSDUCER_MODEL_H_
