// sherpa-onnx/csrc/pad-sequence-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/pad-sequence.h"

#include <numeric>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

TEST(PadSequence, ThreeTensors) {
  Ort::AllocatorWithDefaultOptions allocator;

  std::array<int64_t, 2> shape1{3, 5};
  Ort::Value v1 =
      Ort::Value::CreateTensor<float>(allocator, shape1.data(), shape1.size());
  float *p1 = v1.GetTensorMutableData<float>();
  std::iota(p1, p1 + shape1[0] * shape1[1], 0);

  std::array<int64_t, 2> shape2{4, 5};
  Ort::Value v2 =
      Ort::Value::CreateTensor<float>(allocator, shape2.data(), shape2.size());
  float *p2 = v2.GetTensorMutableData<float>();
  std::iota(p2, p2 + shape2[0] * shape2[1], 0);

  std::array<int64_t, 2> shape3{2, 5};
  Ort::Value v3 =
      Ort::Value::CreateTensor<float>(allocator, shape3.data(), shape3.size());
  float *p3 = v3.GetTensorMutableData<float>();
  std::iota(p3, p3 + shape3[0] * shape3[1], 0);

  auto ans = PadSequence(allocator, {&v1, &v2, &v3}, -1);

  Print2D(&v1);
  Print2D(&v2);
  Print2D(&v3);
  Print3D(&ans);
}

}  // namespace sherpa_onnx
