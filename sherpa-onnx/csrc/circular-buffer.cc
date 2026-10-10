// sherpa-onnx/csrc/circular-buffer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/circular-buffer.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

CircularBuffer::CircularBuffer(int32_t capacity) {
  if (capacity <= 0) {
    SHERPA_ONNX_LOGE("Please specify a positive capacity. Given: %d\n",
                     capacity);
    SHERPA_ONNX_EXIT(-1);
  }
  buffer_.resize(capacity);
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
    begin_ = 0;
    return;
  }

  std::vector<float> new_buffer(new_capacity);
  int32_t part1_size = std::min(size, capacity - begin_);
  std::copy(buffer_.begin() + begin_, buffer_.begin() + begin_ + part1_size,
            new_buffer.begin());
  std::copy(buffer_.begin(), buffer_.begin() + size - part1_size,
            new_buffer.begin() + part1_size);

  buffer_.swap(new_buffer);
  begin_ = 0;
}

void CircularBuffer::Push(const float *p, int32_t n) {
  if (n < 0) {
    SHERPA_ONNX_LOGE("Invalid n: %d", n);
    return;
  }

  if (n == 0) {
    return;
  }

  if (!p) {
    SHERPA_ONNX_LOGE("p is NULL");
    return;
  }

  int32_t capacity = static_cast<int32_t>(buffer_.size());
  int32_t size = Size();
  int64_t required_size = static_cast<int64_t>(n) + size;
  if (required_size > std::numeric_limits<int32_t>::max()) {
    SHERPA_ONNX_LOGE("n + size exceeds INT32_MAX. n: %d, size: %d", n, size);
    return;
  }

  if (required_size > capacity) {
    int32_t new_capacity = static_cast<int32_t>(std::max(
        std::min(static_cast<int64_t>(capacity) * 2,
                 static_cast<int64_t>(std::numeric_limits<int32_t>::max())),
        required_size));
#if __OHOS__
    SHERPA_ONNX_LOGE(
        "Overflow! n: %{public}d, size: %{public}d, n+size: %{public}d, "
        "capacity: %{public}d. Increase "
        "capacity to: %{public}d. (Original data is copied. No data loss!)",
        n, size, static_cast<int32_t>(required_size), capacity, new_capacity);
#else
    SHERPA_ONNX_LOGE(
        "Overflow! n: %d, size: %d, n+size: %d, capacity: %d. Increase "
        "capacity to: %d. (Original data is copied. No data loss!)",
        n, size, static_cast<int32_t>(required_size), capacity, new_capacity);
#endif
    Resize(new_capacity);

    capacity = new_capacity;
  }

  int32_t start =
      static_cast<int32_t>((static_cast<int64_t>(begin_) + size_) % capacity);
  size_ = static_cast<int32_t>(required_size);

  if (n <= capacity - start) {
    std::copy(p, p + n, buffer_.begin() + start);
    return;
  }

  int32_t part1_size = capacity - start;

  std::copy(p, p + part1_size, buffer_.begin() + start);

  std::copy(p + part1_size, p + n, buffer_.begin());
}

int32_t CircularBuffer::GetIndex(int32_t offset) const {
  if (offset < 0 || offset > size_) {
    SHERPA_ONNX_LOGE("Invalid offset: %d. size: %d", offset, size_);
    return Head();
  }

  uint32_t index = head_index_ + static_cast<uint32_t>(offset);
  if (index >= kIndexRange) {
    index -= kIndexRange;
  }

  return static_cast<int32_t>(index);
}

std::vector<float> CircularBuffer::Get(int32_t start_index, int32_t n) const {
  uint32_t offset = kIndexRange;
  if (start_index >= 0) {
    uint32_t index = static_cast<uint32_t>(start_index);
    offset = index >= head_index_ ? index - head_index_
                                  : kIndexRange - head_index_ + index;
  }

  if (offset >= static_cast<uint32_t>(size_)) {
    SHERPA_ONNX_LOGE("Invalid start_index: %d. head_: %d, tail_: %d",
                     start_index, Head(), Tail());
    return {};
  }

  int32_t size = Size();
  if (n < 0 || n > size) {
    SHERPA_ONNX_LOGE("Invalid n: %d. size: %d", n, size);
    return {};
  }

  if (static_cast<uint32_t>(n) > static_cast<uint32_t>(size) - offset) {
    SHERPA_ONNX_LOGE("Invalid start_index: %d and n: %d. head_: %d, size: %d",
                     start_index, n, Head(), size);
    return {};
  }

  int32_t capacity = static_cast<int32_t>(buffer_.size());
  int32_t start =
      static_cast<int32_t>((static_cast<int64_t>(begin_) + offset) % capacity);

  if (n <= capacity - start) {
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

  if (n == 0) {
    return;
  }

  int32_t capacity = static_cast<int32_t>(buffer_.size());
  begin_ = static_cast<int32_t>((static_cast<int64_t>(begin_) + n) % capacity);

  head_index_ += static_cast<uint32_t>(n);
  if (head_index_ >= kIndexRange) {
    head_index_ -= kIndexRange;
  }

  size_ -= n;
  if (size_ == 0) {
    begin_ = 0;
  }
}

}  // namespace sherpa_onnx
