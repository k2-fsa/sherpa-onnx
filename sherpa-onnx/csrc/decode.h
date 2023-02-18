// sherpa/csrc/decode.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_DECODE_H_
#define SHERPA_ONNX_CSRC_DECODE_H_

#include <vector>

#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

/** Greedy search for non-streaming ASR.
 *
 * @TODO(fangjun) Support batch size > 1
 *
 * @param model  The RnntModel
 * @param encoder_out  Its shape is (1, num_frames, encoder_out_dim).
 */
void GreedySearch(OnlineTransducerModel *model, Ort::Value encoder_out,
                  std::vector<int64_t> *hyp);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_DECODE_H_
