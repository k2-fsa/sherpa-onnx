// sherpa-onnx/csrc/slice.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/slice.h"

#include <assert.h>

#include <vector>

namespace sherpa_onnx {

template <typename T /*=float*/>
Ort::Value Slice(const Ort::Value *v, int32_t dim0, int32_t dim1_start,
                 int32_t dim1_end) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  assert(shape.size() == 3);

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::array<int64_t, 2> ans_shape{dim1_end - dim1_start, shape[2]};
  const T *src = v->GetTensorData<T>();
  src += dim0 * shape[1] * shape[2] + dim1_start * shape[2];

  return Ort::Value::CreateTensor(memory_info, const_cast<T *>(src),
                                  ans_shape[0] * ans_shape[1], ans_shape.data(),
                                  ans_shape.size());
}

template Ort::Value Slice<float>(const Ort::Value *v, int32_t dim0,
                                 int32_t dim1_start, int32_t dim1_end);

}  // namespace sherpa_onnx
