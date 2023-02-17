/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sherpa-onnx/csrc/rnnt-model.h"

#include <array>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

RnntModel::RnntModel(const std::string &encoder_filename,
                     const std::string &decoder_filename,
                     const std::string &joiner_filename, int32_t num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING) {
  sess_opts_.SetIntraOpNumThreads(num_threads);
  sess_opts_.SetInterOpNumThreads(num_threads);

  InitEncoder(encoder_filename);
  InitDecoder(decoder_filename);
  InitJoiner(joiner_filename);
}

void RnntModel::InitEncoder(const std::string &filename) {
  encoder_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);
  GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                &encoder_input_names_ptr_);

  GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                 &encoder_output_names_ptr_);
}

void RnntModel::InitDecoder(const std::string &filename) {
  decoder_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                &decoder_input_names_ptr_);

  GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                 &decoder_output_names_ptr_);
}

void RnntModel::InitJoiner(const std::string &filename) {
  joiner_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                &joiner_input_names_ptr_);

  GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                 &joiner_output_names_ptr_);
}

Ort::Value RnntModel::RunEncoder(const float *features, int32_t T,
                                 int32_t feature_dim) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 3> x_shape{1, T, feature_dim};
  Ort::Value x =
      Ort::Value::CreateTensor(memory_info, const_cast<float *>(features),
                               T * feature_dim, x_shape.data(), x_shape.size());

  // std::array<int64_t, 1> x_lens_shape{1};
  // int64_t x_lens_tmp = T;

  // Ort::Value x_lens = Ort::Value::CreateTensor(
  //     memory_info, &x_lens_tmp, 1, x_lens_shape.data(), x_lens_shape.size());

  std::array<Ort::Value, 1> encoder_inputs{std::move(x)};

  // Note: We discard encoder_out_lens since we only implement
  // batch==1.
  auto encoder_out = encoder_sess_->Run(
      {}, encoder_input_names_ptr_.data(), encoder_inputs.data(),
      encoder_inputs.size(), encoder_output_names_ptr_.data(),
      encoder_output_names_ptr_.size());
  return std::move(encoder_out[0]);
}

Ort::Value RnntModel::RunDecoder(const int64_t *decoder_input,
                                 int32_t context_size) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  int32_t batch_size = 1;  // TODO(fangjun): handle the case when it's > 1
  std::array<int64_t, 2> shape{batch_size, context_size};
  Ort::Value in = Ort::Value::CreateTensor(
      memory_info, const_cast<int64_t *>(decoder_input),
      batch_size * context_size, shape.data(), shape.size());

  auto decoder_out = decoder_sess_->Run(
      {}, decoder_input_names_ptr_.data(), &in, 1,
      decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
  return std::move(decoder_out[0]);
}

Ort::Value RnntModel::RunJoiner(const float *projected_encoder_out,
                                const float *projected_decoder_out,
                                int32_t joiner_dim) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  int32_t batch_size = 1;  // TODO(fangjun): handle the case when it's > 1
  std::array<int64_t, 2> shape{batch_size, joiner_dim};

  Ort::Value enc = Ort::Value::CreateTensor(
      memory_info, const_cast<float *>(projected_encoder_out),
      batch_size * joiner_dim, shape.data(), shape.size());

  Ort::Value dec = Ort::Value::CreateTensor(
      memory_info, const_cast<float *>(projected_decoder_out),
      batch_size * joiner_dim, shape.data(), shape.size());

  std::array<Ort::Value, 2> inputs{std::move(enc), std::move(dec)};

  auto logit = joiner_sess_->Run(
      {}, joiner_input_names_ptr_.data(), inputs.data(), inputs.size(),
      joiner_output_names_ptr_.data(), joiner_output_names_ptr_.size());

  return std::move(logit[0]);
}

}  // namespace sherpa_onnx
