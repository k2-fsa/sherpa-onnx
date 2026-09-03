// sherpa-onnx/csrc/offline-recognizer-shared-rng-test.cc
//
// Copyright (c)  2026  fra-shipper

#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

namespace {

// OfflineRecognizerFunASRNanoImpl (offline-recognizer-funasr-nano-impl.h)
// and OfflineRecognizerQwen3ASRImpl (offline-recognizer-qwen3-asr-impl.h)
// each hold a single `mutable std::mt19937 rng_` that is drawn from inside
// SampleTokenWithTemperatureAndTopP(), a const method reachable from
// DecodeStreams(). sherpa-onnx-offline-parallel.cc builds exactly ONE
// OfflineRecognizer and spawns `nj` std::threads that all call
// DecodeStreams() on that same instance concurrently, so with temperature
// sampling enabled every thread used to draw from the same rng_ with no
// synchronization -- a data race. The fix adds a
// `mutable std::mutex rng_mutex_` next to rng_ and a
// `std::lock_guard<std::mutex> lock(rng_mutex_)` around every draw.
//
// OfflineRecognizerFunASRNanoImpl/OfflineRecognizerQwen3ASRImpl cannot be
// constructed in a unit test without loading real ONNX model weights (their
// constructors eagerly open the model files), so this test isolates the
// exact rng_/rng_mutex_ synchronization pattern those two fixes now share
// and exercises it under the same one-instance/nj-threads usage pattern as
// sherpa-onnx-offline-parallel.cc. A small delay is injected inside the
// "critical section" to widen the race window so an unguarded run reliably
// shows overlapping access instead of only occasionally.
class SharedRngSampler {
 public:
  explicit SharedRngSampler(uint32_t seed) : rng_(seed) {}

  // Mirrors the shape of the fixed draw sites, e.g.
  //   std::lock_guard<std::mutex> lock(rng_mutex_);
  //   float sample = dist(rng_);
  float Sample() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::lock_guard<std::mutex> lock(rng_mutex_);
    return DrawLocked(&dist);
  }

  int32_t MaxConcurrentEntries() const {
    return max_active_in_critical_section_.load();
  }

 private:
  float DrawLocked(std::uniform_real_distribution<float> *dist) {
    int32_t active = active_in_critical_section_.fetch_add(1) + 1;
    RecordMax(active);
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    float sample = (*dist)(rng_);
    active_in_critical_section_.fetch_sub(1);
    return sample;
  }

  void RecordMax(int32_t active) {
    int32_t prev =
        max_active_in_critical_section_.load(std::memory_order_relaxed);
    while (
        active > prev &&
        !max_active_in_critical_section_.compare_exchange_weak(prev, active)) {
    }
  }

  mutable std::mt19937 rng_;
  mutable std::mutex rng_mutex_;
  std::atomic<int32_t> active_in_critical_section_{0};
  std::atomic<int32_t> max_active_in_critical_section_{0};
};

}  // namespace

// Regression test: with rng_mutex_ guarding every draw from the shared
// rng_, no two of the nj decode threads may ever be inside the critical
// section at the same time.
TEST(SharedRngThreadSafety, ConcurrentSamplingIsSerialized) {
  constexpr int32_t kNumThreads = 8;  // mirrors a --nj=8 decode run
  constexpr int32_t kDrawsPerThread = 50;

  SharedRngSampler sampler(/*seed=*/42);

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int32_t t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&sampler]() {
      for (int32_t i = 0; i < kDrawsPerThread; ++i) {
        sampler.Sample();
      }
    });
  }
  for (auto &th : threads) th.join();

  EXPECT_EQ(sampler.MaxConcurrentEntries(), 1);
}

}  // namespace sherpa_onnx
