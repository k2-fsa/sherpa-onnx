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

#ifndef SHERPA_ONNX_CSRC_DECODE_H_
#define SHERPA_ONNX_CSRC_DECODE_H_

#include <vector>

#include "sherpa-onnx/csrc/rnnt-model.h"

namespace sherpa_onnx {

/** Greedy search for non-streaming ASR.
 *
 * @TODO(fangjun) Support batch size > 1
 *
 * @param model  The RnntModel
 * @param encoder_out  Its shape is (1, num_frames, encoder_out_dim).
 */
std::vector<int32_t> GreedySearch(RnntModel &model,  // NOLINT
                                  const Ort::Value &encoder_out);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_DECODE_H_
