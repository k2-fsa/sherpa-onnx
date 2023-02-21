// sherpa-onnx/csrc/unbind.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_UNBIND_H_
#define SHERPA_ONNX_CSRC_UNBIND_H_

#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

/** It is similar to torch.unbind() but we keep the unbind dim to 1 in
 * the output
 *
 * @param allocator Allocator to allocate space for the returned tensor
 * @param value  The tensor to unbind
 * @param dim  The dim along which to unbind the tensor
 *
 * @return Return a list of tensors
 */
template <typename T = float>
std::vector<Ort::Value> Unbind(OrtAllocator *allocator, const Ort::Value *value,
                               int32_t dim);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_UNBIND_H_
