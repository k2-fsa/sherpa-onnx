// sherpa-onnx/csrc/offline-whisper-dtw.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_DTW_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_DTW_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace sherpa_onnx {

// Result of DTW alignment
struct DTWResult {
  std::vector<int32_t> text_indices;  // Token index at each alignment point
  std::vector<int32_t> time_indices;  // Frame index at each alignment point
};

// Token timing result from DTW
struct TokenTimingResult {
  std::vector<float> start_times;   // Start time in seconds for each token
  std::vector<float> durations;     // Duration in seconds for each token
};

// Class for processing cross-attention weights and computing DTW alignment
// for token-level timestamps in Whisper.
//
// Based on OpenAI Whisper (whisper/timing.py) and whisper.cpp implementations.
class WhisperDTW {
 public:
  // Compute token timings (start times and durations) from raw cross-attention.
  // This follows OpenAI's approach of extracting both start and end times
  // directly from DTW jump_times, where:
  //   start_times[i] = jump_times[i]
  //   end_times[i] = jump_times[i+1]
  //   durations[i] = end_times[i] - start_times[i]
  //
  // @param attention Raw attention weights from decoder.
  //                  Shape: (n_heads, n_tokens, n_audio_frames)
  // @param n_heads Number of alignment heads
  // @param n_tokens Number of text tokens (including SOT sequence and EOT)
  // @param n_frames Number of audio frames (full context, e.g., 1500)
  // @param num_audio_frames Actual audio frames to use (for clipping)
  // @param sot_sequence_length Number of special tokens at start (to skip)
  // @param num_text_tokens Number of actual text tokens to return timings for
  //                        (excluding SOT sequence and EOT)
  // @param timestamp_token_indices Indices of timestamp tokens to filter out
  //                                (0-based, relative to attention sequence)
  //
  // @return TokenTimingResult with start_times and durations for each token
  TokenTimingResult ComputeTokenTimings(
      const float *attention, int32_t n_heads, int32_t n_tokens,
      int32_t n_frames, int32_t num_audio_frames, int32_t sot_sequence_length,
      int32_t num_text_tokens,
      const std::vector<int32_t> &timestamp_token_indices = {});

 private:
  // Apply softmax normalization across the last dimension (frames)
  void ApplySoftmax(float *data, int32_t n_tokens, int32_t n_frames);

  // Apply z-score normalization across tokens (dim=-2)
  void ApplyZScoreNormalization(float *data, int32_t n_tokens, int32_t n_frames);

  // Apply median filter across frames with given width
  void ApplyMedianFilter(float *data, int32_t n_tokens, int32_t n_frames,
                         int32_t width = 7);

  // Run DTW algorithm on cost matrix
  //
  // @param cost_matrix Negated alignment matrix (n_tokens, n_frames)
  //                    Lower values = better alignment
  // @param n_tokens Number of rows (text tokens)
  // @param n_frames Number of columns (audio frames)
  //
  // @return DTW alignment path
  DTWResult RunDTW(const float *cost_matrix, int32_t n_tokens,
                   int32_t n_frames);
};

// Time conversion constant: 50 tokens per second (20ms per token/frame)
constexpr float kWhisperTokensPerSecond = 50.0f;
constexpr float kWhisperSecondsPerToken = 0.02f;

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_DTW_H_
