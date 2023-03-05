// sherpa-onnx/csrc/slice.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SLICE_H_
#define SHERPA_ONNX_CSRC_SLICE_H_

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

/** Get a shallow copy by slicing v.
 *
 * It returns v[dim0, dim1_start:dim1_end]
 *
 * @param v A 3-D tensor. Its data type is T.
 * @param dim0  Start index of the first dimension..
 * @param dim1_start Start index of the second dimension.
 * @param dim1_end  End index of the second dimension.
 *
 * @return Return a 2-D tensor of shape (dim1_end-dim1_start, v.shape[2])
 *
 * @caution: The returned tensor is a shallow copy of `v`!
 */
template <typename T = float>
Ort::Value Slice(const Ort::Value *v, int32_t dim0, int32_t dim1_start,
                 int32_t dim1_end);
}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SLICE_H_
