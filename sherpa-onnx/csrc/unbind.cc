// sherpa-onnx/csrc/unbind.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/unbind.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

template <typename T /*= float*/>
std::vector<Ort::Value> Unbind(OrtAllocator *allocator, const Ort::Value *value,
                               int32_t dim) {
  std::vector<int64_t> shape = value->GetTensorTypeAndShapeInfo().GetShape();
  assert(dim >= 0);
  assert(dim < static_cast<int32_t>(shape.size()));
  int32_t n = static_cast<int32_t>(shape[dim]);
  if (n == 1) {
    std::vector<Ort::Value> ans;
    ans.push_back(Clone(allocator, value));
    return ans;
  }

  std::vector<int64_t> ans_shape = shape;
  ans_shape[dim] = 1;  // // Unlike torch, we keep the dim to 1

  // allocator tensors
  std::vector<Ort::Value> ans;
  ans.reserve(n);
  for (int32_t i = 0; i != n; ++i) {
    Ort::Value t = Ort::Value::CreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
    ans.push_back(std::move(t));
  }

  auto leading_size = static_cast<int32_t>(std::accumulate(
      shape.begin(), shape.begin() + dim, 1, std::multiplies<int64_t>()));

  auto trailing_size = static_cast<int32_t>(std::accumulate(
      shape.begin() + dim + 1, shape.end(), 1, std::multiplies<int64_t>()));

  const T *src = value->GetTensorData<T>();

  for (int32_t i = 0; i != leading_size; ++i) {
    for (int32_t k = 0; k != n; ++k) {
      T *dst = ans[k].GetTensorMutableData<T>() + i * trailing_size;
      std::copy(src, src + trailing_size, dst);
      src += trailing_size;
    }
  }

  return ans;
}

template std::vector<Ort::Value> Unbind<float>(OrtAllocator *allocator,
                                               const Ort::Value *value,
                                               int32_t dim);

template std::vector<Ort::Value> Unbind<int64_t>(OrtAllocator *allocator,
                                                 const Ort::Value *value,
                                                 int32_t dim);

}  // namespace sherpa_onnx
