// sherpa-onnx/csrc/transpose-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/transpose.h"

#include <numeric>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

TEST(Tranpose, Tranpose01) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 3> shape{3, 2, 5};
  Ort::Value v =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v.GetTensorMutableData<float>();

  std::iota(p, p + shape[0] * shape[1] * shape[2], 0);

  auto ans = Transpose01(allocator, &v);
  auto v2 = Transpose01(allocator, &ans);

  Print3D(&v);
  Print3D(&ans);
  Print3D(&v2);

  const float *q = v2.GetTensorData<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    EXPECT_EQ(p[i], q[i]);
  }
}

TEST(Tranpose, Tranpose12) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 3> shape{3, 2, 5};
  Ort::Value v =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v.GetTensorMutableData<float>();

  std::iota(p, p + shape[0] * shape[1] * shape[2], 0);

  auto ans = Transpose12(allocator, &v);
  auto v2 = Transpose12(allocator, &ans);

  Print3D(&v);
  Print3D(&ans);
  Print3D(&v2);

  const float *q = v2.GetTensorData<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    EXPECT_EQ(p[i], q[i]);
  }
}

}  // namespace sherpa_onnx
