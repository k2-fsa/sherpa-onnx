// sherpa-onnx/csrc/circular-buffer.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_CIRCULAR_BUFFER_H_
#define SHERPA_ONNX_CSRC_CIRCULAR_BUFFER_H_

#include <cstdint>
#include <limits>
#include <vector>

namespace sherpa_onnx {

class CircularBuffer {
 public:
  // Initial capacity of this buffer. It grows when needed.
  explicit CircularBuffer(int32_t capacity);

  // Push an array
  //
  // @param p Pointer to the start address of the array
  // @param n Number of elements in the array
  //
  // Note: The buffer grows if n + Size() is greater than its capacity.
  void Push(const float *p, int32_t n);

  // @param start_index Should be in the range [Head(), Tail())
  // @param n Number of elements to get
  // @return Return a vector of size n containing the requested elements
  std::vector<float> Get(int32_t start_index, int32_t n) const;

  // Return the logical index at the given offset from Head().
  // @param offset Should be in the range [0, Size()]
  int32_t GetIndex(int32_t offset) const;

  // Remove n elements from the buffer
  //
  // @param n Should be in the range [0, size_]
  void Pop(int32_t n);

  // Number of elements in the buffer.
  int32_t Size() const { return size_; }

  // Current logical position of the head.
  // It wraps to 0 after reaching INT32_MAX.
  int32_t Head() const { return static_cast<int32_t>(head_index_); }

  // Current logical position of the tail.
  // It wraps to 0 after reaching INT32_MAX.
  int32_t Tail() const { return GetIndex(size_); }

  void Reset() {
    begin_ = 0;
    size_ = 0;
    head_index_ = 0;
  }

  void Resize(int32_t new_capacity);

 private:
  friend class CircularBuffer_LogicalIndexWrap_Test;

  static constexpr uint32_t kIndexRange =
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1;

  std::vector<float> buffer_;

  int32_t begin_ = 0;  // physical index of the first element
  int32_t size_ = 0;
  uint32_t head_index_ = 0;  // logical index; wraps at kIndexRange
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_CIRCULAR_BUFFER_H_
