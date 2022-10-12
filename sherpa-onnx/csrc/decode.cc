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

#include "sherpa-onnx/csrc/decode.h"

#include <assert.h>

#include <algorithm>
#include <vector>

namespace sherpa_onnx {

std::vector<int32_t> GreedySearch(RnntModel &model,  // NOLINT
                                  const Ort::Value &encoder_out) {
  std::vector<int64_t> encoder_out_shape =
      encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  assert(encoder_out_shape[0] == 1 && "Only batch_size=1 is implemented");
  Ort::Value projected_encoder_out =
      model.RunJoinerEncoderProj(encoder_out.GetTensorData<float>(),
                                 encoder_out_shape[1], encoder_out_shape[2]);

  const float *p_projected_encoder_out =
      projected_encoder_out.GetTensorData<float>();

  int32_t context_size = 2;  // hard-code it to 2
  int32_t blank_id = 0;      // hard-code it to 0
  std::vector<int32_t> hyp(context_size, blank_id);
  std::array<int64_t, 2> decoder_input{blank_id, blank_id};

  Ort::Value decoder_out = model.RunDecoder(decoder_input.data(), context_size);

  std::vector<int64_t> decoder_out_shape =
      decoder_out.GetTensorTypeAndShapeInfo().GetShape();

  Ort::Value projected_decoder_out = model.RunJoinerDecoderProj(
      decoder_out.GetTensorData<float>(), decoder_out_shape[2]);

  int32_t joiner_dim =
      projected_decoder_out.GetTensorTypeAndShapeInfo().GetShape()[1];

  int32_t T = encoder_out_shape[1];
  for (int32_t t = 0; t != T; ++t) {
    Ort::Value logit = model.RunJoiner(
        p_projected_encoder_out + t * joiner_dim,
        projected_decoder_out.GetTensorData<float>(), joiner_dim);

    int32_t vocab_size = logit.GetTensorTypeAndShapeInfo().GetShape()[1];

    const float *p_logit = logit.GetTensorData<float>();

    auto y = static_cast<int32_t>(std::distance(
        static_cast<const float *>(p_logit),
        std::max_element(static_cast<const float *>(p_logit),
                         static_cast<const float *>(p_logit) + vocab_size)));

    if (y != blank_id) {
      decoder_input[0] = hyp.back();
      decoder_input[1] = y;
      hyp.push_back(y);
      decoder_out = model.RunDecoder(decoder_input.data(), context_size);
      projected_decoder_out = model.RunJoinerDecoderProj(
          decoder_out.GetTensorData<float>(), decoder_out_shape[2]);
    }
  }

  return {hyp.begin() + context_size, hyp.end()};
}

}  // namespace sherpa_onnx
