// sherpa-onnx/csrc/circular-buffer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/circular-buffer.h"

#include <algorithm>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

CircularBuffer::CircularBuffer(int32_t capacity) {
  if (capacity <= 0) {
    SHERPA_ONNX_LOGE("Please specify a positive capacity. Given: %d\n",
                     capacity);
    exit(-1);
  }
  buffer_.resize(capacity);
}

void CircularBuffer::Push(const float *p, int32_t n) {
  int32_t capacity = buffer_.size();
  int32_t size = Size();
  if (n + size > capacity) {
    SHERPA_ONNX_LOGE("Overflow! n: %d, size: %d, n+size: %d, capacity: %d", n,
                     size, n + size, capacity);
    exit(-1);
  }

  int32_t start = tail_ % capacity;

  tail_ += n;

  if (start + n < capacity) {
    std::copy(p, p + n, buffer_.begin() + start);
    return;
  }

  int32_t part1_size = capacity - start;

  std::copy(p, p + part1_size, buffer_.begin() + start);

  std::copy(p + part1_size, p + n, buffer_.begin());
}

std::vector<float> CircularBuffer::Get(int32_t start_index, int32_t n) const {
  if (start_index < head_ || start_index >= tail_) {
    SHERPA_ONNX_LOGE("Invalid start_index: %d. head_: %d, tail_: %d",
                     start_index, head_, tail_);
    return {};
  }

  int32_t size = Size();
  if (n < 0 || n > size) {
    SHERPA_ONNX_LOGE("Invalid n: %d. size: %d", n, size);
    return {};
  }

  int32_t capacity = buffer_.size();

  if (start_index - head_ + n > size) {
    SHERPA_ONNX_LOGE("Invalid start_index: %d and n: %d. head_: %d, size: %d",
                     start_index, n, head_, size);
    return {};
  }

  int32_t start = start_index % capacity;

  if (start + n < capacity) {
    return {buffer_.begin() + start, buffer_.begin() + start + n};
  }

  std::vector<float> ans(n);

  std::copy(buffer_.begin() + start, buffer_.end(), ans.begin());

  int32_t part1_size = capacity - start;
  int32_t part2_size = n - part1_size;
  std::copy(buffer_.begin(), buffer_.begin() + part2_size,
            ans.begin() + part1_size);

  return ans;
}

void CircularBuffer::Pop(int32_t n) {
  int32_t size = Size();
  if (n < 0 || n > size) {
    SHERPA_ONNX_LOGE("Invalid n: %d. size: %d", n, size);
    return;
  }

  head_ += n;
}

}  // namespace sherpa_onnx
