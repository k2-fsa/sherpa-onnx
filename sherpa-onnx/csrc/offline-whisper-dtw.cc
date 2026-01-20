// sherpa-onnx/csrc/offline-whisper-dtw.cc
//
// Copyright (c)  2026  Posit Software, PBC

#include "sherpa-onnx/csrc/offline-whisper-dtw.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <cstdio>  // For debug output

// Set to 1 to enable debug output
#define DTW_DEBUG 0

namespace sherpa_onnx {

TokenTimingResult WhisperDTW::ComputeTokenTimings(
    const float *attention, int32_t n_heads, int32_t n_tokens, int32_t n_frames,
    int32_t num_audio_frames, int32_t sot_sequence_length,
    int32_t num_text_tokens,
    const std::vector<int32_t> &timestamp_token_indices) {
  TokenTimingResult result;

  if (n_heads <= 0 || n_tokens <= 0 || n_frames <= 0 || num_text_tokens <= 0) {
    return result;
  }

#if DTW_DEBUG
  fprintf(stderr, "\n========== DTW TIMING DEBUG ==========\n");
  fprintf(stderr, "Input: n_heads=%d, n_tokens=%d, n_frames=%d\n",
          n_heads, n_tokens, n_frames);
  fprintf(stderr, "num_audio_frames=%d, sot_sequence_length=%d, num_text_tokens=%d\n",
          num_audio_frames, sot_sequence_length, num_text_tokens);
  fprintf(stderr, "timestamp_token_indices count: %zu\n",
          timestamp_token_indices.size());
#endif

  // Clip to actual audio frames (like OpenAI: weights[:, :, :num_frames//2])
  int32_t clipped_frames = std::min(n_frames, num_audio_frames);
  if (clipped_frames <= 0) {
    clipped_frames = n_frames;
  }

  // Process attention weights per-head, then average (like OpenAI)
  std::vector<float> processed(n_tokens * clipped_frames, 0.0f);
  std::vector<float> head_data(n_tokens * clipped_frames);

  for (int32_t h = 0; h < n_heads; ++h) {
    const float *src = attention + h * n_tokens * n_frames;
    for (int32_t t = 0; t < n_tokens; ++t) {
      for (int32_t f = 0; f < clipped_frames; ++f) {
        head_data[t * clipped_frames + f] = src[t * n_frames + f];
      }
    }

    ApplySoftmax(head_data.data(), n_tokens, clipped_frames);
    ApplyZScoreNormalization(head_data.data(), n_tokens, clipped_frames);
    ApplyMedianFilter(head_data.data(), n_tokens, clipped_frames, 7);

    for (int32_t i = 0; i < n_tokens * clipped_frames; ++i) {
      processed[i] += head_data[i];
    }
  }

  float inv_n_heads = 1.0f / static_cast<float>(n_heads);
  for (int32_t i = 0; i < n_tokens * clipped_frames; ++i) {
    processed[i] *= inv_n_heads;
  }

  // Build a set of timestamp token indices for quick lookup.
  // The DTW algorithm needs an "anchor" token at position sot_sequence_length
  // to establish the time=0 reference point (like OpenAI's timing.py).
  //
  // Two modes, same anchor position:
  // - enable_segment_timestamps=true: the first timestamp token (e.g. <|0.00|>)
  //   is at index sot_sequence_length. We keep it as the anchor and filter
  //   out subsequent timestamp tokens to avoid alignment drift.
  // - enable_segment_timestamps=false: timestamp_token_indices is empty,
  //   no filtering occurs. But the implementation of enable_segment_timestamps
  //   being false happens to insert a <no_timestamps> token at index
  //   sot_sequence_length, so that will serve as the anchor in this case.
  std::vector<bool> is_timestamp_token(n_tokens, false);
  bool found_first_timestamp = false;
  for (int32_t idx : timestamp_token_indices) {
    if (idx >= 0 && idx < n_tokens) {
      // Keep the first timestamp token (it's the anchor), filter the rest
      if (!found_first_timestamp && idx >= sot_sequence_length) {
        found_first_timestamp = true;
        // Don't mark as timestamp - keep it in the DTW matrix
      } else {
        is_timestamp_token[idx] = true;
      }
    }
  }

  // Skip SOT sequence and filter out timestamp tokens (except first one)
  // Like OpenAI: we skip sot_sequence_length tokens at the start
  // Additionally, we now filter out timestamp tokens from the middle
  int32_t start_token = sot_sequence_length;

  // Build filtered token list (indices into original processed array)
  // and mapping from filtered index back to original index
  std::vector<int32_t> filtered_to_original;
  for (int32_t i = start_token; i < n_tokens; ++i) {
    if (!is_timestamp_token[i]) {
      filtered_to_original.push_back(i);
    }
  }

  int32_t dtw_tokens = static_cast<int32_t>(filtered_to_original.size());

#if DTW_DEBUG
  fprintf(stderr, "DTW tokens after filtering: %d (filtered out %zu timestamp tokens)\n",
          dtw_tokens, timestamp_token_indices.size());
#endif

  if (dtw_tokens <= 1) {
    return result;
  }

  // Extract the filtered portion for DTW and negate
  std::vector<float> cost_matrix(dtw_tokens * clipped_frames);
  for (int32_t i = 0; i < dtw_tokens; ++i) {
    int32_t orig_idx = filtered_to_original[i];
    for (int32_t j = 0; j < clipped_frames; ++j) {
      cost_matrix[i * clipped_frames + j] =
          -processed[orig_idx * clipped_frames + j];
    }
  }

  // Run DTW
  DTWResult dtw_result = RunDTW(cost_matrix.data(), dtw_tokens, clipped_frames);

  if (dtw_result.text_indices.empty()) {
    return result;
  }

  // Extract jump times (where text_idx changes)
  // Like OpenAI: jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1)
  //              jump_times = time_indices[jumps] / TOKENS_PER_SECOND
  std::vector<int32_t> jump_frame_indices;
  jump_frame_indices.push_back(dtw_result.time_indices[0]);  // First is always a jump

  for (size_t i = 1; i < dtw_result.text_indices.size(); ++i) {
    if (dtw_result.text_indices[i] != dtw_result.text_indices[i - 1]) {
      jump_frame_indices.push_back(dtw_result.time_indices[i]);
    }
  }

#if DTW_DEBUG
  fprintf(stderr, "jump_frame_indices count: %zu\n", jump_frame_indices.size());
  fprintf(stderr, "jump_times (first 10): ");
  for (size_t i = 0; i < std::min(size_t(10), jump_frame_indices.size()); ++i) {
    fprintf(stderr, "%.2f ", jump_frame_indices[i] * kWhisperSecondsPerToken);
  }
  fprintf(stderr, "\n");
#endif

  // Now extract start_times and durations for text tokens only (not EOT)
  // Like OpenAI: start_times = jump_times[word_boundaries[:-1]]
  //              end_times = jump_times[word_boundaries[1:]]
  // For tokens (each token is one "word"): boundaries = [0, 1, 2, ..., N]
  // So: start_times[i] = jump_times[i], end_times[i] = jump_times[i+1]
  result.start_times.reserve(num_text_tokens);
  result.durations.reserve(num_text_tokens);

  for (int32_t i = 0; i < num_text_tokens; ++i) {
    if (i < static_cast<int32_t>(jump_frame_indices.size())) {
      float start = static_cast<float>(jump_frame_indices[i]) * kWhisperSecondsPerToken;
      result.start_times.push_back(start);

      // Duration = end_time - start_time = jump_times[i+1] - jump_times[i]
      if (i + 1 < static_cast<int32_t>(jump_frame_indices.size())) {
        float end = static_cast<float>(jump_frame_indices[i + 1]) * kWhisperSecondsPerToken;
        result.durations.push_back(end - start);
      } else {
        // Last token: duration to end of audio
        float audio_end = static_cast<float>(clipped_frames) * kWhisperSecondsPerToken;
        result.durations.push_back(std::max(0.0f, audio_end - start));
      }
    } else {
      // Fallback: use last known time
      float last_time = result.start_times.empty() ? 0.0f : result.start_times.back();
      result.start_times.push_back(last_time);
      result.durations.push_back(0.0f);
    }
  }

#if DTW_DEBUG
  fprintf(stderr, "Result: %zu start_times, %zu durations\n",
          result.start_times.size(), result.durations.size());
  fprintf(stderr, "========== END DTW TIMING DEBUG ==========\n\n");
#endif

  return result;
}

void WhisperDTW::ApplySoftmax(float *data, int32_t n_tokens, int32_t n_frames) {
  for (int32_t t = 0; t < n_tokens; ++t) {
    float *row = data + t * n_frames;

    // Find max for numerical stability
    float max_val = *std::max_element(row, row + n_frames);

    // Compute exp and sum
    float sum = 0.0f;
    for (int32_t f = 0; f < n_frames; ++f) {
      row[f] = std::exp(row[f] - max_val);
      sum += row[f];
    }

    // Normalize
    if (sum > 0.0f) {
      float inv_sum = 1.0f / sum;
      for (int32_t f = 0; f < n_frames; ++f) {
        row[f] *= inv_sum;
      }
    }
  }
}

void WhisperDTW::ApplyZScoreNormalization(float *data, int32_t n_tokens,
                                           int32_t n_frames) {
  // Normalize across tokens (dim=-2) for each frame
  for (int32_t f = 0; f < n_frames; ++f) {
    // Compute mean
    float sum = 0.0f;
    for (int32_t t = 0; t < n_tokens; ++t) {
      sum += data[t * n_frames + f];
    }
    float mean = sum / static_cast<float>(n_tokens);

    // Compute std
    float sq_sum = 0.0f;
    for (int32_t t = 0; t < n_tokens; ++t) {
      float diff = data[t * n_frames + f] - mean;
      sq_sum += diff * diff;
    }
    float std_dev = std::sqrt(sq_sum / static_cast<float>(n_tokens) + 1e-9f);

    // Normalize
    float inv_std = 1.0f / std_dev;
    for (int32_t t = 0; t < n_tokens; ++t) {
      data[t * n_frames + f] = (data[t * n_frames + f] - mean) * inv_std;
    }
  }
}

void WhisperDTW::ApplyMedianFilter(float *data, int32_t n_tokens,
                                    int32_t n_frames, int32_t width) {
  if (width <= 1 || n_frames <= 1) {
    return;
  }

  int32_t half_width = width / 2;
  std::vector<float> temp(n_frames);
  std::vector<float> window(width);

  for (int32_t t = 0; t < n_tokens; ++t) {
    float *row = data + t * n_frames;

    // Copy original row
    std::copy(row, row + n_frames, temp.begin());

    for (int32_t f = 0; f < n_frames; ++f) {
      // Gather window values with reflection padding
      int32_t w_idx = 0;
      for (int32_t k = -half_width; k <= half_width && w_idx < width; ++k) {
        int32_t src_idx = f + k;
        // Reflect at boundaries
        if (src_idx < 0) {
          src_idx = -src_idx;
        } else if (src_idx >= n_frames) {
          src_idx = 2 * n_frames - 2 - src_idx;
        }
        src_idx = std::max(0, std::min(src_idx, n_frames - 1));
        window[w_idx++] = temp[src_idx];
      }

      // Sort and take median
      std::sort(window.begin(), window.begin() + w_idx);
      row[f] = window[w_idx / 2];
    }
  }
}

DTWResult WhisperDTW::RunDTW(const float *cost_matrix, int32_t n_tokens,
                              int32_t n_frames) {
  // DTW algorithm based on whisper.cpp and OpenAI Whisper
  // O(N*M) time and space complexity

  DTWResult result;

  if (n_tokens <= 0 || n_frames <= 0) {
    return result;
  }

  constexpr float kInf = std::numeric_limits<float>::infinity();

  int32_t N = n_tokens;
  int32_t M = n_frames;

  // Cost and trace matrices (N+1 x M+1)
  std::vector<float> cost((N + 1) * (M + 1), kInf);
  std::vector<int32_t> trace((N + 1) * (M + 1), -1);

  auto cost_at = [&](int32_t i, int32_t j) -> float& {
    return cost[i * (M + 1) + j];
  };
  auto trace_at = [&](int32_t i, int32_t j) -> int32_t& {
    return trace[i * (M + 1) + j];
  };

  // Initialize
  cost_at(0, 0) = 0.0f;

  // Fill cost matrix
  for (int32_t j = 1; j <= M; ++j) {
    for (int32_t i = 1; i <= N; ++i) {
      float c0 = cost_at(i - 1, j - 1);  // diagonal
      float c1 = cost_at(i - 1, j);      // up
      float c2 = cost_at(i, j - 1);      // left

      float min_cost;
      int32_t trace_dir;

      if (c0 <= c1 && c0 <= c2) {
        min_cost = c0;
        trace_dir = 0;  // diagonal
      } else if (c1 <= c0 && c1 <= c2) {
        min_cost = c1;
        trace_dir = 1;  // up
      } else {
        min_cost = c2;
        trace_dir = 2;  // left
      }

      // Add current cost
      cost_at(i, j) = cost_matrix[(i - 1) * M + (j - 1)] + min_cost;
      trace_at(i, j) = trace_dir;
    }
  }

  // Backtrace
  int32_t i = N;
  int32_t j = M;

  // Force horizontal movement at row 0 and vertical at column 0
  for (int32_t jj = 0; jj <= M; ++jj) {
    trace_at(0, jj) = 2;  // left
  }
  for (int32_t ii = 0; ii <= N; ++ii) {
    trace_at(ii, 0) = 1;  // up
  }

  std::vector<std::pair<int32_t, int32_t>> path;
  path.reserve(N + M);

  while (i > 0 || j > 0) {
    path.push_back({i - 1, j - 1});

    int32_t dir = trace_at(i, j);
    if (dir == 0) {  // diagonal
      --i;
      --j;
    } else if (dir == 1) {  // up
      --i;
    } else {  // left
      --j;
    }
  }

  // Reverse path (we built it backwards)
  std::reverse(path.begin(), path.end());

  // Extract result
  result.text_indices.reserve(path.size());
  result.time_indices.reserve(path.size());

  for (const auto &p : path) {
    if (p.first >= 0 && p.second >= 0) {
      result.text_indices.push_back(p.first);
      result.time_indices.push_back(p.second);
    }
  }

  return result;
}

}  // namespace sherpa_onnx
