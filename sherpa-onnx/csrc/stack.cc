// sherpa-onnx/csrc/stack.cc
//
// Copyright (c) 2023 Jingzhao Ou (jingzhao.ou@gmail.com)

#include "sherpa-onnx/csrc/stack.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

static bool Compare(const std::vector<int64_t> &a,
                    const std::vector<int64_t> &b) {
  if (a.size() != b.size()) return false;

  for (int32_t i = 0; i != static_cast<int32_t>(a.size()); ++i) {
    if (a[i] != b[i]) return false;
  }

  return true;
}

static void PrintShape(const std::vector<int64_t> &a) {
  for (auto i : a) {
    fprintf(stderr, "%d ", static_cast<int32_t>(i));
  }
  fprintf(stderr, "\n");
}

template <typename T /*=float*/>
Ort::Value Stack(OrtAllocator *allocator,
                 const std::vector<const Ort::Value *> &values, int32_t dim) {
  std::vector<int64_t> v0_shape =
      values[0]->GetTensorTypeAndShapeInfo().GetShape();

  for (int32_t i = 1; i != static_cast<int32_t>(values.size()); ++i) {
    auto s = values[i]->GetTensorTypeAndShapeInfo().GetShape();
    bool ret = Compare(v0_shape, s);
    if (!ret) {
      fprintf(stderr, "Incorrect shape in Stack !\n");

      fprintf(stderr, "Shape for tensor 0: ");
      PrintShape(v0_shape);

      fprintf(stderr, "Shape for tensor %d: ", i);
      PrintShape(s);

      exit(-1);
    }
  }

  std::vector<int64_t> ans_shape;
  ans_shape.reserve(v0_shape.size() + 1);
  ans_shape.insert(ans_shape.end(), v0_shape.data(), v0_shape.data() + dim);
  ans_shape.push_back(values.size());
  ans_shape.insert(ans_shape.end(), v0_shape.data() + dim,
                   v0_shape.data() + v0_shape.size());

  auto leading_size = static_cast<int32_t>(std::accumulate(
      v0_shape.begin(), v0_shape.begin() + dim, 1, std::multiplies<int64_t>()));

  auto trailing_size = static_cast<int32_t>(std::accumulate(
      v0_shape.begin() + dim, v0_shape.end(), 1, std::multiplies<int64_t>()));

  Ort::Value ans = Ort::Value::CreateTensor<T>(allocator, ans_shape.data(),
                                               ans_shape.size());
  T *dst = ans.GetTensorMutableData<T>();

  for (int32_t i = 0; i != leading_size; ++i) {
    for (auto value : values) {
      const T *src = value->GetTensorData<T>();
      src += i * trailing_size;

      std::copy(src, src + trailing_size, dst);
      dst += trailing_size;
    }
  }

  return ans;
}

template Ort::Value Stack<float>(OrtAllocator *allocator,
                                 const std::vector<const Ort::Value *> &values,
                                 int32_t dim);

template Ort::Value Stack<int64_t>(
    OrtAllocator *allocator, const std::vector<const Ort::Value *> &values,
    int32_t dim);

}  // namespace sherpa_onnx
