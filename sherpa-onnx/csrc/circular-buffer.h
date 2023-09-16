// sherpa-onnx/csrc/circular-buffer.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_CIRCULAR_BUFFER_H_
#define SHERPA_ONNX_CSRC_CIRCULAR_BUFFER_H_

#include <vector>

namespace sherpa_onnx {

class CircularBuffer {
 public:
  // Capacity of this buffer. Should be large enough.
  // If it is full, we just print a message and exit the program.
  explicit CircularBuffer(int32_t capacity);

  // Push an array
  //
  // @param p Pointer to the start address of the array
  // @param n Number of elements in the array
  void Push(const float *p, int32_t n);

  // @param start_index Should in the range [head_, tail_)
  // @param n Number of elements to get
  std::vector<float> Get(int32_t start_index, int32_t n) const;

  // Remove n elements from the buffer
  //
  // @param n Should be in the range [0, size_]
  void Pop(int32_t n);

  int32_t Size() const { return size_; }
  int32_t Head() const { return head_; }
  int32_t Tail() const { return tail_; }

 private:
  std::vector<float> buffer_;

  int32_t size_ = 0;  // number of elements currently in the buffer.
                      // Should never exceed buffer_.size();
                      // Otherwise, it prints an error message and exits the
                      // program.

  int32_t head_ = 0;  // linear index; always increasing; never wraps around
  int32_t tail_ = 0;  // linear index, always increasing; never wraps around.
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_CIRCULAR_BUFFER_H_
