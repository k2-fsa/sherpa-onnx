// sherpa-onnx/csrc/math-test.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/math.h"

#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(Transpose, Case1) {
  // 0 1 2
  // 3 4 5
  std::vector<float> in = {0, 1, 2, 3, 4, 5};
  std::vector<float> out = Transpose(in.data(), 2, 3);

  // 0 3
  // 1 4
  // 2 5
  std::vector<float> expected_out = {0, 3, 1, 4, 2, 5};
  EXPECT_EQ(out, expected_out);
}

TEST(Transpose, Case2) {
  // 0 1
  // 2 3
  // 4 5
  std::vector<float> in = {0, 1, 2, 3, 4, 5};
  std::vector<float> out = Transpose(in.data(), 3, 2);

  // 0 2 4
  // 1 3 5
  std::vector<float> expected_out = {0, 2, 4, 1, 3, 5};
  EXPECT_EQ(out, expected_out);
}

TEST(ScaleAdd, Case1) {
  std::vector<float> src = {1, 2, 3};
  float scale = 10;
  std::vector<float> in_out = {5, 6, 0};
  ScaleAdd(src.data(), scale, src.size(), in_out.data());

  std::vector<float> expected = {10 + 5, 20 + 6, 30 + 0};
  EXPECT_EQ(in_out, expected);
}

TEST(Scale, Case1) {
  std::vector<float> src = {1, 2, 3};
  float scale = 10;
  std::vector<float> in_out = {5, 6, 0};
  Scale(src.data(), scale, src.size(), in_out.data());

  std::vector<float> expected = {10, 20, 30};
  EXPECT_EQ(in_out, expected);
}

TEST(Scale, Case2InPlace) {
  std::vector<float> src = {1, 2, 3};
  float scale = 10;
  Scale(src.data(), scale, src.size(), src.data());

  std::vector<float> expected = {10, 20, 30};
  EXPECT_EQ(src, expected);
}

}  // namespace sherpa_onnx
