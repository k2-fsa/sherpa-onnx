// sherpa-onnx/csrc/packed-sequence-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/packed-sequence.h"

#include <numeric>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

TEST(PackedSequence, Case1) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 3> shape{5, 5, 4};
  Ort::Value v =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = v.GetTensorMutableData<float>();

  std::iota(p, p + shape[0] * shape[1] * shape[2], 0);

  Ort::Value length =
      Ort::Value::CreateTensor<int64_t>(allocator, shape.data(), 1);
  int64_t *p_length = length.GetTensorMutableData<int64_t>();
  p_length[0] = 1;
  p_length[1] = 2;
  p_length[2] = 3;
  p_length[3] = 5;
  p_length[4] = 2;

  auto packed_seq = PackPaddedSequence(allocator, &v, &length);
  fprintf(stderr, "sorted indexes: ");
  for (auto i : packed_seq.sorted_indexes) {
    fprintf(stderr, "%d ", static_cast<int32_t>(i));
  }
  fprintf(stderr, "\n");
  // output index:   0 1 2 3 4
  // sorted indexes: 3 2 1 4 0
  // length:         5 3 2 2 1
  Print3D(&v);
  Print2D(&packed_seq.data);
  fprintf(stderr, "batch sizes per time step: ");
  for (auto i : packed_seq.batch_sizes) {
    fprintf(stderr, "%d ", static_cast<int32_t>(i));
  }
  fprintf(stderr, "\n");

  // TODO(fangjun): Check that the return value is correct
}

}  // namespace sherpa_onnx
