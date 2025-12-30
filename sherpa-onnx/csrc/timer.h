// sherpa-onnx/csrc/timer.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_TIMER_H_
#define SHERPA_ONNX_CSRC_TIMER_H_

#include <memory>

namespace sherpa_onnx {

class Timer {
 public:
  Timer();
  ~Timer();

  void Reset() const;

  // Return time in seconds
  double Elapsed() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TIMER_H_
