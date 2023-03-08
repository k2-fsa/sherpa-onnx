// sherpa-onnx/csrc/packed-sequence.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_PACKED_SEQUENCE_H_
#define SHERPA_ONNX_CSRC_PACKED_SEQUENCE_H_

#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

struct PackedSequence {
  std::vector<int32_t> sorted_indexes;
  std::vector<int32_t> batch_sizes;
  Ort::Value data{nullptr};
};

/** Similar to torch.nn.utils.rnn.pad_sequence but it supports only
 * batch_first=true.
 *
 * @param allocator
 * @param value  A 3-D tensor of shape (B, T, C). Its dtype is float.
 * @param length A 1-D tensor of shape (B,). Its dtype is int64_t. Each
 *               element in it specifies the valid length of the corresponding
 *               entry in value before padding.
 */
PackedSequence PackPaddedSequence(OrtAllocator *allocator,
                                  const Ort::Value *value, Ort::Value *length);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PACKED_SEQUENCE_H_
