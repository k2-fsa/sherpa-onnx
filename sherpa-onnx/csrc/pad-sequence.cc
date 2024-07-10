// sherpa-onnx/csrc/pad-sequence.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/pad-sequence.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace sherpa_onnx {

Ort::Value PadSequence(OrtAllocator *allocator,
                       const std::vector<const Ort::Value *> &values,
                       float padding_value) {
  int32_t batch_size = static_cast<int32_t>(values.size());

  std::vector<int64_t> shape0 =
      values[0]->GetTensorTypeAndShapeInfo().GetShape();
  assert(shape0.size() == 2);

  auto feature_dim = shape0[1];
  auto max_T = shape0[0];

  for (int32_t i = 1; i != batch_size; ++i) {
    auto shape = values[i]->GetTensorTypeAndShapeInfo().GetShape();

    assert(shape.size() == 2);
    assert(shape[1] == feature_dim);

    max_T = std::max(max_T, shape[0]);
  }
  std::array<int64_t, 3> ans_shape{batch_size, max_T, feature_dim};

  Ort::Value ans = Ort::Value::CreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());
  float *dst = ans.GetTensorMutableData<float>();
  std::fill(dst, dst + batch_size * max_T * feature_dim, padding_value);

  for (const auto *v : values) {
    const float *src = v->GetTensorData<float>();
    auto shape = v->GetTensorTypeAndShapeInfo().GetShape();
    std::copy(src, src + shape[0] * shape[1], dst);
    dst += max_T * feature_dim;
  }

  return ans;

  // TODO(fangjun): Check that the returned value is correct.
}

}  // namespace sherpa_onnx
