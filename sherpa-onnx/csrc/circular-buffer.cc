// sherpa-onnx/csrc/circular-buffer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/circular-buffer.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

CircularBuffer::CircularBuffer(int32_t capacity, int64_t initial_position) {
  if (capacity <= 0) {
    SHERPA_ONNX_LOGE("Please specify a positive capacity. Given: %d\n",
                     capacity);
    SHERPA_ONNX_EXIT(-1);
  }
  if (initial_position < 0) {
    // A negative head_/tail_ makes `x % capacity` negative, which would make
    // std::copy() below write to an invalid iterator.
    SHERPA_ONNX_LOGE(
        "Please specify a non-negative initial_position. Given: %lld\n",
        static_cast<long long>(initial_position));
    SHERPA_ONNX_EXIT(-1);
  }
  buffer_.resize(capacity);
  head_ = initial_position;
  tail_ = initial_position;
}

void CircularBuffer::Resize(int32_t new_capacity) {
  int32_t capacity = static_cast<int32_t>(buffer_.size());
  if (new_capacity <= capacity) {
#if __OHOS__
    SHERPA_ONNX_LOGE(
        "new_capacity (%{public}d) <= original capacity (%{public}d). Skip it.",
        new_capacity, capacity);
#else
    SHERPA_ONNX_LOGE("new_capacity (%d) <= original capacity (%d). Skip it.",
                     new_capacity, capacity);
#endif
    return;
  }

  int32_t size = Size();
  if (size == 0) {
    buffer_.resize(new_capacity);
    return;
  }

  std::vector<float> new_buffer(new_capacity);
  int32_t start = static_cast<int32_t>(head_ % capacity);
  int32_t dest = static_cast<int32_t>(head_ % new_capacity);

  if (start + size <= capacity) {
    if (dest + size <= new_capacity) {
      std::copy(buffer_.begin() + start, buffer_.begin() + start + size,
                new_buffer.begin() + dest);
    } else {
      int32_t part1_size = new_capacity - dest;

      // copy [start, start+part1_size] to new_buffer
      std::copy(buffer_.begin() + start, buffer_.begin() + start + part1_size,
                new_buffer.begin() + dest);

      // copy [start+part1_size, start+size] to new_buffer
      std::copy(buffer_.begin() + start + part1_size,
                buffer_.begin() + start + size, new_buffer.begin());
    }
  } else {
    int32_t part1_size = capacity - start;
    int32_t part2_size = size - part1_size;

    // copy [start, start+part1_size] to new_buffer
    if (dest + part1_size <= new_capacity) {
      std::copy(buffer_.begin() + start, buffer_.begin() + start + part1_size,
                new_buffer.begin() + dest);
    } else {
      int32_t first_part = new_capacity - dest;
      std::copy(buffer_.begin() + start, buffer_.begin() + start + first_part,
                new_buffer.begin() + dest);

      std::copy(buffer_.begin() + start + first_part,
                buffer_.begin() + start + part1_size, new_buffer.begin());
    }

    int32_t new_dest = (dest + part1_size) % new_capacity;

    if (new_dest + part2_size <= new_capacity) {
      std::copy(buffer_.begin(), buffer_.begin() + part2_size,
                new_buffer.begin() + new_dest);
    } else {
      int32_t first_part = new_capacity - new_dest;
      std::copy(buffer_.begin(), buffer_.begin() + first_part,
                new_buffer.begin() + new_dest);
      std::copy(buffer_.begin() + first_part, buffer_.begin() + part2_size,
                new_buffer.begin());
    }
  }
  buffer_.swap(new_buffer);
}

void CircularBuffer::Push(const float *p, int32_t n) {
  int32_t capacity = static_cast<int32_t>(buffer_.size());
  int32_t size = Size();
  if (n + size > capacity) {
    int32_t new_capacity = std::max(capacity * 2, n + size);
#if __OHOS__
    SHERPA_ONNX_LOGE(
        "Overflow! n: %{public}d, size: %{public}d, n+size: %{public}d, "
        "capacity: %{public}d. Increase "
        "capacity to: %{public}d. (Original data is copied. No data loss!)",
        n, size, n + size, capacity, new_capacity);
#else
    SHERPA_ONNX_LOGE(
        "Overflow! n: %d, size: %d, n+size: %d, capacity: %d. Increase "
        "capacity to: %d. (Original data is copied. No data loss!)",
        n, size, n + size, capacity, new_capacity);
#endif
    Resize(new_capacity);

    capacity = new_capacity;
  }

  int32_t start = static_cast<int32_t>(tail_ % capacity);

  tail_ += n;

  if (start + n < capacity) {
    std::copy(p, p + n, buffer_.begin() + start);
    return;
  }

  int32_t part1_size = capacity - start;

  std::copy(p, p + part1_size, buffer_.begin() + start);

  std::copy(p + part1_size, p + n, buffer_.begin());
}

std::vector<float> CircularBuffer::Get(int64_t start_index, int32_t n) const {
  if (start_index < head_ || start_index >= tail_) {
    SHERPA_ONNX_LOGE("Invalid start_index: %lld. head_: %lld, tail_: %lld",
                     static_cast<long long>(start_index),
                     static_cast<long long>(head_),
                     static_cast<long long>(tail_));
    return {};
  }

  int32_t size = Size();
  if (n < 0 || n > size) {
    SHERPA_ONNX_LOGE("Invalid n: %d. size: %d", n, size);
    return {};
  }

  int32_t capacity = static_cast<int32_t>(buffer_.size());

  if (start_index - head_ + n > size) {
    SHERPA_ONNX_LOGE(
        "Invalid start_index: %lld and n: %d. head_: %lld, size: %d",
        static_cast<long long>(start_index), n,
        static_cast<long long>(head_), size);
    return {};
  }

  int32_t start = static_cast<int32_t>(start_index % capacity);

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
