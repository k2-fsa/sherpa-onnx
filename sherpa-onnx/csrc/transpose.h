// sherpa-onnx/csrc/transpose.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_TRANSPOSE_H_
#define SHERPA_ONNX_CSRC_TRANSPOSE_H_

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {
/** Transpose a 3-D tensor from shape (B, T, C) to (T, B, C).
 *
 * @param allocator
 * @param v A 3-D tensor of shape (B, T, C). Its data type is type.
 *
 * @return Return a 3-D tensor of shape (T, B, C). Its data type is type.
 */
template <typename type = float>
Ort::Value Transpose01(OrtAllocator *allocator, const Ort::Value *v);

/** Transpose a 3-D tensor from shape (B, T, C) to (B, C, T).
 *
 * @param allocator
 * @param v A 3-D tensor of shape (B, T, C). Its data type is type.
 *
 * @return Return a 3-D tensor of shape (B, C, T). Its data type is type.
 */
template <typename type = float>
Ort::Value Transpose12(OrtAllocator *allocator, const Ort::Value *v);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TRANSPOSE_H_
