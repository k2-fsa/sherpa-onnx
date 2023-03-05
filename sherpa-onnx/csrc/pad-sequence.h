// sherpa-onnx/csrc/pad-sequence.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_PAD_SEQUENCE_H_
#define SHERPA_ONNX_CSRC_PAD_SEQUENCE_H_

#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

/** Similar to torch.nn.utils.rnn.pad_sequence but it supports only
 * batch_first=true.
 *
 * @param allocator
 * @param values A list of 2-D tensors. Each tensor's second dimension
 *               must be the same and the data type of each tensor should
 *               be float.
 * @param padding_value Value used for padding. For log-fbank, you usually use
 *                      -23.025850929940457f as the padding value.
 *
 * @return Return a 3-D tensor of shape (B, max_T, C).
 */
Ort::Value PadSequence(OrtAllocator *allocator,
                       const std::vector<const Ort::Value *> &values,
                       float padding_value);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PAD_SEQUENCE_H_
