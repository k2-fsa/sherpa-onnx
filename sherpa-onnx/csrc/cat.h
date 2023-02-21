// sherpa-onnx/csrc/cat.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_CAT_H_
#define SHERPA_ONNX_CSRC_CAT_H_

#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

/** Cat a list of tensors along the given dim.
 *
 * @param allocator Allocator to allocate space for the returned tensor
 * @param values  Pointer to a list of tensors. The shape of the tensor must
 *                be the same except on the dim to be concatenated.
 * @param dim  The dim along which to concatenate the input tensors
 *
 * @return Return the concatenated tensor
 */
template <typename T = float>
Ort::Value Cat(OrtAllocator *allocator,
               const std::vector<const Ort::Value *> &values, int32_t dim);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_CAT_H_
