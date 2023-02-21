// sherpa-onnx/csrc/unbind-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/unbind.h"

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

TEST(Ubind, Test1DTensors) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 1> shape{3};
  Ort::Value v =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v.GetTensorMutableData<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 0);
  EXPECT_EQ(ans.size(), shape[0]);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    EXPECT_EQ(ans[i].GetTensorData<float>()[0], p[i]);
  }
  Print1D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    Print1D(&ans[i]);
  }

  // For Cat
  std::vector<const Ort::Value *> vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  Ort::Value v2 = Cat(allocator, vec, 0);
  const float *p2 = v2.GetTensorData<float>();
  for (int32_t i = 0; i != shape[0]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test2DTensorsDim0) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 2> shape{3, 2};
  Ort::Value v =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v.GetTensorMutableData<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1]); ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 0);

  Print2D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    Print2D(&ans[i]);
  }

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    const float *pans = ans[i].GetTensorData<float>();
    for (int32_t k = 0; k != static_cast<int32_t>(shape[1]); ++k, ++p) {
      EXPECT_EQ(*p, pans[k]);
    }
  }

  // For Cat
  std::vector<const Ort::Value *> vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  Ort::Value v2 = Cat(allocator, vec, 0);
  Print2D(&v2);

  p = v.GetTensorMutableData<float>();
  const float *p2 = v2.GetTensorData<float>();
  for (int32_t i = 0; i != shape[0] * shape[1]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test2DTensorsDim1) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 2> shape{3, 2};
  Ort::Value v =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v.GetTensorMutableData<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1]); ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 1);

  Print2D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[1]); ++i) {
    Print2D(&ans[i]);
  }

  // For Cat
  std::vector<const Ort::Value *> vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  Ort::Value v2 = Cat(allocator, vec, 1);
  Print2D(&v2);

  p = v.GetTensorMutableData<float>();
  const float *p2 = v2.GetTensorData<float>();
  for (int32_t i = 0; i != shape[0] * shape[1]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test3DTensorsDim0) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 3> shape{3, 2, 5};
  Ort::Value v =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v.GetTensorMutableData<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 0);

  Print3D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    Print3D(&ans[i]);
  }

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    const float *pans = ans[i].GetTensorData<float>();
    for (int32_t k = 0; k != static_cast<int32_t>(shape[1] * shape[2]);
         ++k, ++p) {
      EXPECT_EQ(*p, pans[k]);
    }
  }

  // For Cat
  std::vector<const Ort::Value *> vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  Ort::Value v2 = Cat(allocator, vec, 0);
  Print3D(&v2);

  p = v.GetTensorMutableData<float>();
  const float *p2 = v2.GetTensorData<float>();
  for (int32_t i = 0; i != shape[0] * shape[1] * shape[2]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test3DTensorsDim1) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 3> shape{3, 2, 5};
  Ort::Value v =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v.GetTensorMutableData<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 1);

  Print3D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[1]); ++i) {
    Print3D(&ans[i]);
  }

  // For Cat
  std::vector<const Ort::Value *> vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  Ort::Value v2 = Cat(allocator, vec, 1);
  Print3D(&v2);

  p = v.GetTensorMutableData<float>();
  const float *p2 = v2.GetTensorData<float>();
  for (int32_t i = 0; i != shape[0] * shape[1] * shape[2]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

TEST(Ubind, Test3DTensorsDim2) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 3> shape{3, 2, 5};
  Ort::Value v =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v.GetTensorMutableData<float>();

  for (int32_t i = 0; i != static_cast<int32_t>(shape[0] * shape[1] * shape[2]);
       ++i) {
    p[i] = i;
  }
  auto ans = Unbind(allocator, &v, 2);

  Print3D(&v);
  for (int32_t i = 0; i != static_cast<int32_t>(shape[2]); ++i) {
    Print3D(&ans[i]);
  }

  // For Cat
  std::vector<const Ort::Value *> vec(ans.size());
  for (int32_t i = 0; i != static_cast<int32_t>(vec.size()); ++i) {
    vec[i] = &ans[i];
  }
  Ort::Value v2 = Cat(allocator, vec, 2);
  Print3D(&v2);

  p = v.GetTensorMutableData<float>();
  const float *p2 = v2.GetTensorData<float>();
  for (int32_t i = 0; i != shape[0] * shape[1] * shape[2]; ++i) {
    EXPECT_EQ(p[i], p2[i]);
  }
}

}  // namespace sherpa_onnx
