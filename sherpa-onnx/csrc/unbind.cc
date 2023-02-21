// sherpa-onnx/csrc/unbind.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/unbind.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

std::vector<Ort::Value> Unbind(OrtAllocator *allocator, const Ort::Value *value,
                               int32_t dim) {
  std::vector<int64_t> shape = value->GetTensorTypeAndShapeInfo().GetShape();
  assert(dim >= 0);
  assert(dim < static_cast<int32_t>(shape.size()));
  int32_t n = static_cast<int32_t>(shape[dim]);

  std::vector<int64_t> ans_shape = shape;
  ans_shape[dim] = 1;  // // Unlike torch, we keep the dim to 1

  // allocator tensors
  std::vector<Ort::Value> ans;
  ans.reserve(n);
  for (int32_t i = 0; i != n; ++i) {
    Ort::Value t = Ort::Value::CreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());
    ans.push_back(std::move(t));
  }

  auto leading_size = static_cast<int32_t>(std::accumulate(
      shape.begin(), shape.begin() + dim, 1, std::multiplies<int64_t>()));

  auto trailing_size = static_cast<int32_t>(std::accumulate(
      shape.begin() + dim + 1, shape.end(), 1, std::multiplies<int64_t>()));

  const float *src = value->GetTensorData<float>();

  for (int32_t i = 0; i != leading_size; ++i) {
    for (int32_t k = 0; k != n; ++k) {
      float *dst = ans[k].GetTensorMutableData<float>() + i * trailing_size;
      std::copy(src, src + trailing_size, dst);
      src += trailing_size;
    }
  }

  return std::move(ans);
}

}  // namespace sherpa_onnx
