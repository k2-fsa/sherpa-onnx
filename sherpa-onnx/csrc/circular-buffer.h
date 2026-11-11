// sherpa-onnx/csrc/circular-buffer.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_CIRCULAR_BUFFER_H_
#define SHERPA_ONNX_CSRC_CIRCULAR_BUFFER_H_

#include <cstdint>
#include <vector>

namespace sherpa_onnx {

class CircularBuffer {
 public:
  // Capacity of this buffer. Should be large enough.
  // If it is full, we just print a message and exit the program.
  //
  // @param initial_position Starting value of the sample counter. Must be
  //                         >= 0. It is used by tests to exercise positions
  //                         near the 32-bit boundary. Keep the default 0 in
  //                         normal use.
  explicit CircularBuffer(int32_t capacity, int64_t initial_position = 0);

  // Push an array
  //
  // @param p Pointer to the start address of the array
  // @param n Number of elements in the array
  //
  // Note: If n + Size() > capacity, we print an error message and exit.
  void Push(const float *p, int32_t n);

  // @param start_index Should in the range [head_, tail_)
  // @param n Number of elements to get
  // @return Return a vector of size n containing the requested elements
  std::vector<float> Get(int64_t start_index, int32_t n) const;

  // Remove n elements from the buffer
  //
  // @param n Should be in the range [0, size_]
  void Pop(int32_t n);

  // Number of elements in the buffer.
  // It is bounded by the capacity, so int32_t is wide enough.
  int32_t Size() const { return static_cast<int32_t>(tail_ - head_); }

  // Current position of the head
  int64_t Head() const { return head_; }

  // Current position of the tail
  int64_t Tail() const { return tail_; }

  void Reset() {
    head_ = 0;
    tail_ = 0;
  }

  void Resize(int32_t new_capacity);

 private:
  std::vector<float> buffer_;

  // Linear indices; always increasing; never wrap around.
  // 64-bit so a long-running stream cannot overflow. At 16 kHz, 2^63 samples
  // is about 18 million years.
  int64_t head_ = 0;  // where to read next
  int64_t tail_ = 0;  // where to write next
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_CIRCULAR_BUFFER_H_
