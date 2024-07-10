// sherpa-onnx/csrc/packed-sequence.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/packed-sequence.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <utility>

#include "sherpa-onnx/csrc/slice.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

static Ort::Value IndexSelect(OrtAllocator *allocator, const Ort::Value *value,
                              const std::vector<int32_t> &sorted_indexes) {
  auto shape = value->GetTensorTypeAndShapeInfo().GetShape();
  assert(shape.size() == 3);
  std::array<int64_t, 3> ans_shape{static_cast<int64_t>(sorted_indexes.size()),
                                   shape[1], shape[2]};

  Ort::Value ans = Ort::Value::CreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());
  float *dst = ans.GetTensorMutableData<float>();
  const float *src = value->GetTensorData<float>();

  for (auto i : sorted_indexes) {
    const float *start = src + i * shape[1] * shape[2];
    std::copy(start, start + shape[1] * shape[2], dst);
    dst += shape[1] * shape[2];
  }
  return ans;
}

PackedSequence PackPaddedSequence(OrtAllocator *allocator,
                                  const Ort::Value *value, Ort::Value *length) {
  std::vector<int64_t> v_shape = value->GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> l_shape = length->GetTensorTypeAndShapeInfo().GetShape();

  assert(v_shape.size() == 3);
  assert(l_shape.size() == 1);
  assert(v_shape[0] == l_shape[0]);

  std::vector<int32_t> indexes(v_shape[0]);
  std::iota(indexes.begin(), indexes.end(), 0);

  const int64_t *p_length = length->GetTensorData<int64_t>();
  // sort in descending order
  std::sort(indexes.begin(), indexes.end(), [p_length](int32_t i, int32_t j) {
    return p_length[i] > p_length[j];
  });

  int32_t n = static_cast<int32_t>(v_shape[0]);

  int64_t max_T = p_length[indexes[0]];

  auto sum_T = std::accumulate(p_length, p_length + n, static_cast<int64_t>(0));

  std::array<int64_t, 2> data_shape{sum_T, v_shape[2]};

  Ort::Value data = Ort::Value::CreateTensor<float>(
      allocator, data_shape.data(), data_shape.size());
  float *dst = data.GetTensorMutableData<float>();

  Ort::Value tensor = IndexSelect(allocator, value, indexes);
  tensor = Transpose01(allocator, &tensor);

  // batch size at each time step
  std::vector<int32_t> batch_sizes;
  batch_sizes.reserve(max_T);

  int64_t prev_l = 0;
  for (int32_t i = 0; i != n; ++i) {
    auto cur_l = p_length[indexes[n - 1 - i]];
    assert(cur_l >= prev_l);
    if (cur_l == prev_l) {
      continue;
    }

    auto cur_batch_size = n - i;

    Ort::Value cur_batch =
        Slice(allocator, &tensor, prev_l, cur_l, 0, cur_batch_size);
    auto count = cur_batch.GetTensorTypeAndShapeInfo().GetElementCount();
    const float *src = cur_batch.GetTensorData<float>();
    std::copy(src, src + count, dst);
    dst += count;

    for (int32_t j = prev_l; j < cur_l; ++j) {
      batch_sizes.push_back(cur_batch_size);
    }

    prev_l = cur_l;
  }

  PackedSequence packed_seq;
  packed_seq.sorted_indexes = std::move(indexes);
  packed_seq.data = std::move(data);
  packed_seq.batch_sizes = std::move(batch_sizes);

  return packed_seq;
}

}  // namespace sherpa_onnx
