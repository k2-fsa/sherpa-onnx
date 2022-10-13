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

#ifdef _MSC_VER
// For ToWide() below
#include <codecvt>
#include <locale>
#endif

namespace sherpa_onnx {

#ifdef _MSC_VER
// See
// https://stackoverflow.com/questions/2573834/c-convert-string-or-char-to-wstring-or-wchar-t
static std::wstring ToWide(const std::string &s) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.from_bytes(s);
}
#define SHERPA_MAYBE_WIDE(s) ToWide(s)
#else
#define SHERPA_MAYBE_WIDE(s) s
#endif

/**
 * Get the input names of a model.
 *
 * @param sess An onnxruntime session.
 * @param input_names. On return, it contains the input names of the model.
 * @param input_names_ptr. On return, input_names_ptr[i] contains
 *                         input_names[i].c_str()
 */
static void GetInputNames(Ort::Session *sess,
                          std::vector<std::string> *input_names,
                          std::vector<const char *> *input_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetInputCount();
  input_names->resize(node_count);
  input_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    auto tmp = sess->GetInputNameAllocated(i, allocator);
    (*input_names)[i] = tmp.get();
    (*input_names_ptr)[i] = (*input_names)[i].c_str();
  }
}

/**
 * Get the output names of a model.
 *
 * @param sess An onnxruntime session.
 * @param output_names. On return, it contains the output names of the model.
 * @param output_names_ptr. On return, output_names_ptr[i] contains
 *                         output_names[i].c_str()
 */
static void GetOutputNames(Ort::Session *sess,
                           std::vector<std::string> *output_names,
                           std::vector<const char *> *output_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetOutputCount();
  output_names->resize(node_count);
  output_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    auto tmp = sess->GetOutputNameAllocated(i, allocator);
    (*output_names)[i] = tmp.get();
    (*output_names_ptr)[i] = (*output_names)[i].c_str();
  }
}

RnntModel::RnntModel(const std::string &encoder_filename,
                     const std::string &decoder_filename,
                     const std::string &joiner_filename,
                     const std::string &joiner_encoder_proj_filename,
                     const std::string &joiner_decoder_proj_filename,
                     int32_t num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING) {
  sess_opts_.SetIntraOpNumThreads(num_threads);
  sess_opts_.SetInterOpNumThreads(num_threads);

  InitEncoder(encoder_filename);
  InitDecoder(decoder_filename);
  InitJoiner(joiner_filename);
  InitJoinerEncoderProj(joiner_encoder_proj_filename);
  InitJoinerDecoderProj(joiner_decoder_proj_filename);
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

void RnntModel::InitJoinerEncoderProj(const std::string &filename) {
  joiner_encoder_proj_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(joiner_encoder_proj_sess_.get(),
                &joiner_encoder_proj_input_names_,
                &joiner_encoder_proj_input_names_ptr_);

  GetOutputNames(joiner_encoder_proj_sess_.get(),
                 &joiner_encoder_proj_output_names_,
                 &joiner_encoder_proj_output_names_ptr_);
}

void RnntModel::InitJoinerDecoderProj(const std::string &filename) {
  joiner_decoder_proj_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(joiner_decoder_proj_sess_.get(),
                &joiner_decoder_proj_input_names_,
                &joiner_decoder_proj_input_names_ptr_);

  GetOutputNames(joiner_decoder_proj_sess_.get(),
                 &joiner_decoder_proj_output_names_,
                 &joiner_decoder_proj_output_names_ptr_);
}

Ort::Value RnntModel::RunEncoder(const float *features, int32_t T,
                                 int32_t feature_dim) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 3> x_shape{1, T, feature_dim};
  Ort::Value x =
      Ort::Value::CreateTensor(memory_info, const_cast<float *>(features),
                               T * feature_dim, x_shape.data(), x_shape.size());

  std::array<int64_t, 1> x_lens_shape{1};
  int64_t x_lens_tmp = T;

  Ort::Value x_lens = Ort::Value::CreateTensor(
      memory_info, &x_lens_tmp, 1, x_lens_shape.data(), x_lens_shape.size());

  std::array<Ort::Value, 2> encoder_inputs{std::move(x), std::move(x_lens)};

  // Note: We discard encoder_out_lens since we only implement
  // batch==1.
  auto encoder_out = encoder_sess_->Run(
      {}, encoder_input_names_ptr_.data(), encoder_inputs.data(),
      encoder_inputs.size(), encoder_output_names_ptr_.data(),
      encoder_output_names_ptr_.size());
  return std::move(encoder_out[0]);
}
Ort::Value RnntModel::RunJoinerEncoderProj(const float *encoder_out, int32_t T,
                                           int32_t encoder_out_dim) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::array<int64_t, 2> in_shape{T, encoder_out_dim};
  Ort::Value in = Ort::Value::CreateTensor(
      memory_info, const_cast<float *>(encoder_out), T * encoder_out_dim,
      in_shape.data(), in_shape.size());

  auto encoder_proj_out = joiner_encoder_proj_sess_->Run(
      {}, joiner_encoder_proj_input_names_ptr_.data(), &in, 1,
      joiner_encoder_proj_output_names_ptr_.data(),
      joiner_encoder_proj_output_names_ptr_.size());
  return std::move(encoder_proj_out[0]);
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

Ort::Value RnntModel::RunJoinerDecoderProj(const float *decoder_out,
                                           int32_t decoder_out_dim) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  int32_t batch_size = 1;  // TODO(fangjun): handle the case when it's > 1
  std::array<int64_t, 2> shape{batch_size, decoder_out_dim};
  Ort::Value in = Ort::Value::CreateTensor(
      memory_info, const_cast<float *>(decoder_out),
      batch_size * decoder_out_dim, shape.data(), shape.size());

  auto decoder_proj_out = joiner_decoder_proj_sess_->Run(
      {}, joiner_decoder_proj_input_names_ptr_.data(), &in, 1,
      joiner_decoder_proj_output_names_ptr_.data(),
      joiner_decoder_proj_output_names_ptr_.size());
  return std::move(decoder_proj_out[0]);
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
