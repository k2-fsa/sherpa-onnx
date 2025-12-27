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

// Class for processing cross-attention weights and computing DTW alignment
// for word-level timestamps in Whisper.
//
// Based on OpenAI Whisper (whisper/timing.py) and whisper.cpp implementations.
class WhisperDTW {
 public:
  // Compute alignment from raw cross-attention weights.
  //
  // @param attention Raw attention weights from decoder.
  //                  Shape: (n_heads, n_tokens, n_audio_frames)
  // @param n_heads Number of alignment heads
  // @param n_tokens Number of text tokens
  // @param n_frames Number of audio frames (full context, e.g., 1500)
  // @param num_audio_frames Actual audio frames to use (for clipping)
  // @param sot_sequence_length Number of special tokens at start (to skip)
  //
  // @return Frame index for each text token (excluding SOT sequence)
  std::vector<int32_t> ComputeAlignment(const float *attention, int32_t n_heads,
                                        int32_t n_tokens, int32_t n_frames,
                                        int32_t num_audio_frames,
                                        int32_t sot_sequence_length = 3);

  // Get timestamps in seconds for each token.
  // Each audio frame represents 20ms (TOKENS_PER_SECOND = 50).
  //
  // @param frame_indices Frame index for each token
  // @return Timestamp in seconds for each token
  static std::vector<float> FrameIndicesToSeconds(
      const std::vector<int32_t> &frame_indices);

 private:
  // Process attention weights:
  // 1. Average across heads
  // 2. Apply softmax normalization (across frames)
  // 3. Apply z-score normalization (across tokens)
  // 4. Apply median filter (across frames)
  //
  // @param attention Input attention weights (n_heads, n_tokens, n_frames)
  // @param output Output processed matrix (n_tokens, n_frames)
  void ProcessAttention(const float *attention, float *output, int32_t n_heads,
                        int32_t n_tokens, int32_t n_frames);

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

// Word boundary information
struct WhisperWordBoundary {
  std::string word;       // The word text
  int32_t start_token;    // Index of first token in this word
  int32_t end_token;      // Index of last token in this word (exclusive)
};

// Compute word boundaries from decoded tokens.
// Groups tokens into words based on space prefixes and CJK character detection.
//
// @param tokens Vector of decoded token strings
// @return Vector of word boundaries
std::vector<WhisperWordBoundary> ComputeWordBoundaries(
    const std::vector<std::string> &tokens);

// Check if a string starts with a space character
bool StartsWithSpace(const std::string &s);

// Check if a character is CJK (Chinese, Japanese, Korean)
bool IsCJKCharacter(uint32_t codepoint);

// Decode UTF-8 and get the first codepoint
uint32_t GetFirstCodepoint(const std::string &s);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_DTW_H_
