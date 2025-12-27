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

/*

import numpy as np

def compute_mean_and_inv_std(p: np.ndarray):
    mean = p.mean(axis=0)
    var = np.maximum((p**2).mean(axis=0) - mean**2, 0.0)
    std = np.sqrt(var)
    inv_std = 1.0 / (std + 1e-5)
    return mean.astype(np.float32), inv_std.astype(np.float32)

def dump_cpp_vector(name: str, arr: np.ndarray):
    flat = arr.flatten()
    print(f"std::vector<float> {name} = {{")
    line = ""
    for i, v in enumerate(flat):
        line += f"{v:.8f}f, "
        if (i + 1) % 8 == 0:
            print("  " + line)
            line = ""
    if line:
        print("  " + line)
    print("};\n")

np.random.seed(42)
num_rows, num_cols = 4, 6
x = np.random.randn(num_rows, num_cols).astype(np.float32)

mean, inv_std = compute_mean_and_inv_std(x)

dump_cpp_vector("x", x)
dump_cpp_vector("mean", mean)
dump_cpp_vector("inv_std", inv_std)

 */

TEST(ComputeMeanAndInvStd, Case1) {
  std::vector<float> x = {
      0.49671414f,  -0.13826430f, 0.64768857f, 1.52302980f,  -0.23415338f,
      -0.23413695f, 1.57921278f,  0.76743472f, -0.46947438f, 0.54256004f,
      -0.46341768f, -0.46572974f, 0.24196227f, -1.91328025f, -1.72491789f,
      -0.56228751f, -1.01283109f, 0.31424734f, -0.90802407f, -1.41230369f,
      1.46564877f,  -0.22577630f, 0.06752820f, -1.42474818f,
  };

  std::vector<float> expected_mean = {
      0.35246629f, -0.67410338f, -0.02026373f,
      0.31938151f, -0.41071847f, -0.45259190f,
  };

  std::vector<float> expected_inv_std = {
      1.13103926f, 0.94854516f, 0.83320111f,
      1.24679470f, 2.52932906f, 1.59057319f,
  };

  std::vector<float> mean;
  std::vector<float> inv_std;

  int32_t num_rows = 4;
  int32_t num_cols = 6;

  ComputeMeanAndInvStd(x.data(), num_rows, num_cols, &mean, &inv_std);

  ASSERT_EQ(mean.size(), num_cols);
  ASSERT_EQ(inv_std.size(), num_cols);

  for (int32_t i = 0; i < num_cols; ++i) {
    EXPECT_NEAR(mean[i], expected_mean[i], 1e-6f) << "at index " << i;
    EXPECT_NEAR(inv_std[i], expected_inv_std[i], 1e-6f) << "at index " << i;
  }
}

}  // namespace sherpa_onnx
