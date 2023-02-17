// sherpa/csrc/online-lstm-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-lstm-transducer-model.h"

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

OnlineLstmTransducerModel::OnlineLstmTransducerModel(
    const OnlineTransducerModelConfig &model_config)
    : env_(ORT_LOGGING_LEVEL_WARNING) {
  sess_opts_.SetIntraOpNumThreads(model_config.num_threads);
  sess_opts_.SetInterOpNumThreads(model_config.num_threads);

  InitEncoder(model_config.encoder_filename);
  InitDecoder(model_config.decoder_filename);
  InitJoiner(model_config.joiner_filename);
}

void OnlineLstmTransducerModel::InitEncoder(const std::string &filename) {
  encoder_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                &encoder_input_names_ptr_);

  GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                 &encoder_output_names_ptr_);
}

void OnlineLstmTransducerModel::InitDecoder(const std::string &filename) {
  decoder_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                &decoder_input_names_ptr_);

  GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                 &decoder_output_names_ptr_);
}

void OnlineLstmTransducerModel::InitJoiner(const std::string &filename) {
  joiner_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                &joiner_input_names_ptr_);

  GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                 &joiner_output_names_ptr_);
}

Ort::Value OnlineLstmTransducerModel::StackStates(
    const std::vector<Ort::Value> &states) const {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  int64_t a;
  std::array<int64_t, 3> x_shape{1, 1, 1};
  Ort::Value x = Ort::Value::CreateTensor(memory_info, &a, 0, &a, 0);
  return x;
}

std::vector<Ort::Value> OnlineLstmTransducerModel::UnStackStates(
    Ort::Value states) const {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);
  return {};
  // auto memory_info =
  //     Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  // int64_t a;
  // std::array<int64_t, 3> x_shape{1, 1, 1};
  // std::vector<Ort::Value> b{
  //     Ort::Value::CreateTensor(memory_info, &a, 0, &a, 0)};
  //
  // return b;
}

Ort::Value OnlineLstmTransducerModel::GetEncoderInitStates() {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  int64_t a;
  std::array<int64_t, 3> x_shape{1, 1, 1};
  Ort::Value x = Ort::Value::CreateTensor(memory_info, &a, 0, &a, 0);

  return x;
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineLstmTransducerModel::RunEncoder(const Ort::Value &features,
                                      const std::vector<Ort::Value> &states) {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 3> x_shape{1, 1, 1};
  int64_t a;

  return {Ort::Value::CreateTensor(memory_info, &a, 0, &a, 0),
          std::vector<Ort::Value>{}};
}

Ort::Value OnlineLstmTransducerModel::RunDecoder(
    const Ort::Value &decoder_input) {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  int64_t a = 0;
  std::array<int64_t, 3> x_shape{1, 1, 1};
  Ort::Value x = Ort::Value::CreateTensor(memory_info, &a, 0, &a, 0);

  return x;
}

Ort::Value OnlineLstmTransducerModel::RunJoiner(const Ort::Value &encoder_out,
                                                const Ort::Value &decoder_out) {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 3> x_shape{1, 1, 1};
  int64_t a;
  Ort::Value x = Ort::Value::CreateTensor(memory_info, &a, 0, &a, 0);

  return x;
}

int32_t OnlineLstmTransducerModel::ContextSize() const {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);
  return {};
}

int32_t OnlineLstmTransducerModel::ChunkSize() const {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);
  return {};
}

int32_t OnlineLstmTransducerModel::ChunkShift() const {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);
  return {};
}

int32_t OnlineLstmTransducerModel::VocabSize() const {
  fprintf(stderr, "implement me: %s:%d!\n", __func__, (int)__LINE__);
  return {};
}

}  // namespace sherpa_onnx
