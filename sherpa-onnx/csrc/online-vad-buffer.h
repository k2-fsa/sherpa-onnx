// sherpa-onnx/csrc/online-vad-buffer.h
//
// Copyright (c)  2026  Manickavela Arumugam
//
// Per-stream VAD prediction buffer -- mirrors the FeatureExtractor API so
// that VAD predictions and fbank features are accessed with the same
// indexing convention and the same method names. This is the piece that
// lets OnlineRecognizer::IsReady()'s VAD-coverage gate ask "how far has
// VAD actually scored" without the ASR and VAD pipelines sharing any
// other state.
//
// Backed by CircularBuffer (the same rolling buffer used by
// VoiceActivityDetector).
//
// Thread-safe: a single mutex_ protects all operations, matching the
// pattern used by FeatureExtractor::Impl in features.cc. VAD writes
// (Push) happen on the independent VAD pipeline's thread; ASR reads (At,
// GetFrames, NumFramesReady) happen on the ASR decode thread.
#ifndef SHERPA_ONNX_CSRC_ONLINE_VAD_BUFFER_H_
#define SHERPA_ONNX_CSRC_ONLINE_VAD_BUFFER_H_

#include <cstdint>
#include <mutex>   // NOLINT
#include <vector>

#include "sherpa-onnx/csrc/circular-buffer.h"

namespace sherpa_onnx {

class OnlineVadBuffer {
 public:
  // capacity: max predictions before auto-resize.
  explicit OnlineVadBuffer(int32_t capacity = 8192) : cb_(capacity) {}

  // Append one VAD prediction (1.0f speech / 0.0f silence) to the tail.
  void Push(float prob) {
    std::lock_guard<std::mutex> lock(mutex_);
    cb_.Push(&prob, 1);
  }

  void Pop(int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    cb_.Pop(n);
  }

  // Head-relative read: 0 = oldest available. Returns 0.0f if out of range.
  float At(int32_t i) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (i < 0 || i >= cb_.Size()) return 0.0f;
    auto v = cb_.Get(cb_.Head() + i, 1);
    return v.empty() ? 0.0f : v[0];
  }

  int32_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cb_.Size();
  }

  // Total predictions pushed since last Reset().
  int32_t NumFramesReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cb_.Tail();
  }

  // Plain read, no pop.
  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cb_.Get(frame_index, n);
  }

  // Number of trailing "silence" (prob <= threshold) predictions, i.e.
  // how many predictions from the tail backwards are all silence. Used
  // directly by the endpoint check: trailing_silence_predictions *
  // window_shift_seconds >= min_silence_duration.
  int32_t NumTrailingSilence(float threshold) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int32_t size = cb_.Size();
    if (size == 0) return 0;
    auto v = cb_.Get(cb_.Head(), size);
    int32_t count = 0;
    for (int32_t i = static_cast<int32_t>(v.size()) - 1; i >= 0; --i) {
      if (v[i] > threshold) break;
      ++count;
    }
    return count;
  }

  // True if any prediction ever pushed was above threshold.
  bool EverSpeech(float threshold) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int32_t size = cb_.Size();
    if (size == 0) return false;
    auto v = cb_.Get(cb_.Head(), size);
    for (float p : v) {
      if (p > threshold) return true;
    }
    return false;
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    cb_.Reset();
  }

 private:
  CircularBuffer cb_;
  mutable std::mutex mutex_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_VAD_BUFFER_H_
