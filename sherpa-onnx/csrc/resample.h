/**
 * Copyright     2013  Pegah Ghahremani
 *               2014  IMSL, PKU-HKUST (author: Wei Shi)
 *               2014  Yanqing Sun, Junjie Wang
 *               2014  Johns Hopkins University (author: Daniel Povey)
 * Copyright     2023  Xiaomi Corporation (authors: Fangjun Kuang)
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
// The implementation lives in kaldi-native-fbank; this header only provides
// an alias so existing callers keep working.
#ifndef SHERPA_ONNX_CSRC_RESAMPLE_H_
#define SHERPA_ONNX_CSRC_RESAMPLE_H_

#include "kaldi-native-fbank/csrc/resample.h"

namespace sherpa_onnx {

using LinearResample = knf::LinearResample;

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RESAMPLE_H_
