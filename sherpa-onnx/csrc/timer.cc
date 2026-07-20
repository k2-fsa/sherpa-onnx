// sherpa-onnx/csrc/timer.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/timer.h"

#include <chrono>
#include <memory>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

// modified from https://github.com/kaldi-asr/kaldi/blob/master/src/base/timer.h
class Timer::Impl {
 public:
  explicit Impl(bool debug) : debug_(debug) {
    if (debug_) {
      Reset();
    }
  }

  using high_resolution_clock = std::chrono::high_resolution_clock;

  void Reset() {
    if (!debug_) {
      return;
    }

    begin_ = high_resolution_clock::now();
  }

  // Return time in seconds
  double Elapsed() {
    if (!debug_) {
      return 0;
    }

    auto end = high_resolution_clock::now();
    auto diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin_);
    return diff.count() / 1000000.0;
  }

  void Log(const char *tag) {
    if (!debug_) {
      return;
    }

#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s %{public}.3f s", tag, Elapsed());
#else
    SHERPA_ONNX_LOGE("%s %.3f s", tag, Elapsed());
#endif
  }

 private:
  bool debug_;
  high_resolution_clock::time_point begin_;
};

Timer::Timer(bool debug) : impl_(std::make_unique<Impl>(debug)) {}

Timer::~Timer() = default;

void Timer::Reset() const { impl_->Reset(); }

// Return time in seconds
double Timer::Elapsed() const { return impl_->Elapsed(); }

void Timer::Log(const char *tag) const { impl_->Log(tag); }

}  // namespace sherpa_onnx
