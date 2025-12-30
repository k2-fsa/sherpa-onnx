// sherpa-onnx/csrc/timer.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/timer.h"

#include <chrono>
#include <memory>

namespace sherpa_onnx {

// modified from https://github.com/kaldi-asr/kaldi/blob/master/src/base/timer.h
class Timer::Impl {
 public:
  Impl() { Reset(); }

  using high_resolution_clock = std::chrono::high_resolution_clock;

  void Reset() { begin_ = high_resolution_clock::now(); }

  // Return time in seconds
  double Elapsed() {
    auto end = high_resolution_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin_);
    return diff.count() / 1000000.0;
  }

 private:
  high_resolution_clock::time_point begin_;
};

Timer::Timer() : impl_(std::make_unique<Impl>()) {}

Timer::~Timer() = default;

void Timer::Reset() const { impl_->Reset(); }

// Return time in seconds
double Timer::Elapsed() const { return impl_->Elapsed(); }

}  // namespace sherpa_onnx
