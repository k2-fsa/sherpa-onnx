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

#ifndef SHERPA_ONNX_CSRC_RNNT_MODEL_H_
#define SHERPA_ONNX_CSRC_RNNT_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

class RnntModel {
 public:
  /**
   * @param encoder_filename Path to the encoder model
   * @param decoder_filename Path to the decoder model
   * @param joiner_filename Path to the joiner model
   * @param joiner_encoder_proj_filename Path to the joiner encoder_proj model
   * @param joiner_decoder_proj_filename Path to the joiner decoder_proj model
   * @param num_threads Number of threads to use to run the models
   */
  RnntModel(const std::string &encoder_filename,
            const std::string &decoder_filename,
            const std::string &joiner_filename,
            const std::string &joiner_encoder_proj_filename,
            const std::string &joiner_decoder_proj_filename,
            int32_t num_threads);

  /** Run the encoder model.
   *
   * @TODO(fangjun): Support batch_size > 1
   *
   * @param features  A tensor of shape (batch_size, T, feature_dim)
   * @param  T Number of feature frames
   * @param  feature_dim  Dimension of the feature.
   *
   * @return Return  a tensor of shape (batch_size, T', encoder_out_dim)
   */
  Ort::Value RunEncoder(const float *features, int32_t T, int32_t feature_dim);

  /** Run the joiner encoder_proj model.
   *
   * @param encoder_out A tensor of shape (T, encoder_out_dim)
   * @param T Number of frames in encoder_out.
   * @param encoder_out_dim  Dimension of encoder_out.
   *
   * @return Return a tensor of shape (T, joiner_dim)
   *
   */
  Ort::Value RunJoinerEncoderProj(const float *encoder_out, int32_t T,
                                  int32_t encoder_out_dim);

  /** Run the decoder model.
   *
   * @TODO(fangjun): Support batch_size > 1
   *
   * @param decoder_input  A tensor of shape (batch_size, context_size).
   * @return Return a tensor of shape (batch_size, 1, decoder_out_dim)
   */
  Ort::Value RunDecoder(const int64_t *decoder_input, int32_t context_size);

  /** Run joiner decoder_proj model.
   *
   * @TODO(fangjun): Support batch_size > 1
   *
   * @param decoder_out A tensor of shape (batch_size, decoder_out_dim)
   * @param decoder_out_dim Output dimension of the decoder_out.
   *
   * @return Return a tensor of shape (batch_size, joiner_dim);
   */
  Ort::Value RunJoinerDecoderProj(const float *decoder_out,
                                  int32_t decoder_out_dim);

  /** Run the joiner model.
   *
   * @TODO(fangjun): Support batch_size > 1
   *
   * @param projected_encoder_out  A tensor of shape (batch_size, joiner_dim).
   * @param projected_decoder_out  A tensor of shape (batch_size, joiner_dim).
   *
   * @return Return a tensor of shape (batch_size, vocab_size)
   */
  Ort::Value RunJoiner(const float *projected_encoder_out,
                       const float *projected_decoder_out, int32_t joiner_dim);

 private:
  void InitEncoder(const std::string &encoder_filename);
  void InitDecoder(const std::string &decoder_filename);
  void InitJoiner(const std::string &joiner_filename);
  void InitJoinerEncoderProj(const std::string &joiner_encoder_proj_filename);
  void InitJoinerDecoderProj(const std::string &joiner_decoder_proj_filename);

 private:
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;
  std::unique_ptr<Ort::Session> joiner_sess_;
  std::unique_ptr<Ort::Session> joiner_encoder_proj_sess_;
  std::unique_ptr<Ort::Session> joiner_decoder_proj_sess_;

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

  std::vector<std::string> joiner_encoder_proj_input_names_;
  std::vector<const char *> joiner_encoder_proj_input_names_ptr_;
  std::vector<std::string> joiner_encoder_proj_output_names_;
  std::vector<const char *> joiner_encoder_proj_output_names_ptr_;

  std::vector<std::string> joiner_decoder_proj_input_names_;
  std::vector<const char *> joiner_decoder_proj_input_names_ptr_;
  std::vector<std::string> joiner_decoder_proj_output_names_;
  std::vector<const char *> joiner_decoder_proj_output_names_ptr_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RNNT_MODEL_H_
