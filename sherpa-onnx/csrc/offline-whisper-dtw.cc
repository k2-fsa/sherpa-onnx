// sherpa-onnx/csrc/offline-whisper-dtw.cc
//
// Copyright (c)  2024  Xiaomi Corporation

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

std::vector<int32_t> WhisperDTW::ComputeAlignment(const float *attention,
                                                   int32_t n_heads,
                                                   int32_t n_tokens,
                                                   int32_t n_frames,
                                                   int32_t num_audio_frames,
                                                   int32_t sot_sequence_length) {
  if (n_heads <= 0 || n_tokens <= 0 || n_frames <= 0) {
    return {};
  }

#if DTW_DEBUG
  fprintf(stderr, "\n========== DTW DEBUG ==========\n");
  fprintf(stderr, "Input: n_heads=%d, n_tokens=%d, n_frames=%d\n",
          n_heads, n_tokens, n_frames);
  fprintf(stderr, "num_audio_frames=%d, sot_sequence_length=%d\n",
          num_audio_frames, sot_sequence_length);
#endif

  // Clip to actual audio frames (like OpenAI: weights[:, :, :num_frames//2])
  int32_t clipped_frames = std::min(n_frames, num_audio_frames);
  if (clipped_frames <= 0) {
    clipped_frames = n_frames;  // Fallback if num_audio_frames not set
  }

#if DTW_DEBUG
  fprintf(stderr, "Clipped frames: %d\n", clipped_frames);
#endif

  // Process attention weights per-head, then average (like OpenAI)
  std::vector<float> processed(n_tokens * clipped_frames, 0.0f);
  std::vector<float> head_data(n_tokens * clipped_frames);

  for (int32_t h = 0; h < n_heads; ++h) {
    // Copy and clip this head's data
    const float *src = attention + h * n_tokens * n_frames;
    for (int32_t t = 0; t < n_tokens; ++t) {
      for (int32_t f = 0; f < clipped_frames; ++f) {
        head_data[t * clipped_frames + f] = src[t * n_frames + f];
      }
    }

    // Apply softmax across frames (like OpenAI: .softmax(dim=-1))
    ApplySoftmax(head_data.data(), n_tokens, clipped_frames);

    // Apply z-score normalization across tokens (like OpenAI: (weights - mean) / std)
    ApplyZScoreNormalization(head_data.data(), n_tokens, clipped_frames);

    // Apply median filter (like OpenAI: median_filter(weights, medfilt_width))
    ApplyMedianFilter(head_data.data(), n_tokens, clipped_frames, 7);

    // Accumulate for averaging
    for (int32_t i = 0; i < n_tokens * clipped_frames; ++i) {
      processed[i] += head_data[i];
    }
  }

  // Average across heads (like OpenAI: matrix = weights.mean(axis=0))
  float inv_n_heads = 1.0f / static_cast<float>(n_heads);
  for (int32_t i = 0; i < n_tokens * clipped_frames; ++i) {
    processed[i] *= inv_n_heads;
  }

  // Skip SOT sequence tokens and trailing EOT
  // (like OpenAI: matrix = matrix[len(tokenizer.sot_sequence) : -1])
  int32_t start_token = sot_sequence_length;
  int32_t effective_tokens = n_tokens - sot_sequence_length - 1;  // -1 for EOT

#if DTW_DEBUG
  fprintf(stderr, "After processing: effective_tokens=%d (skip first %d, last 1)\n",
          effective_tokens, start_token);
  // Print min/max of processed matrix
  float pmin = processed[0], pmax = processed[0];
  for (int32_t i = 0; i < n_tokens * clipped_frames; ++i) {
    pmin = std::min(pmin, processed[i]);
    pmax = std::max(pmax, processed[i]);
  }
  fprintf(stderr, "Processed matrix: min=%.6f, max=%.6f\n", pmin, pmax);
#endif

  if (effective_tokens <= 0) {
    return {};
  }

  // Extract the relevant portion for DTW and negate
  // (like OpenAI: text_indices, time_indices = dtw(-matrix))
  std::vector<float> cost_matrix(effective_tokens * clipped_frames);
  for (int32_t i = 0; i < effective_tokens; ++i) {
    for (int32_t j = 0; j < clipped_frames; ++j) {
      cost_matrix[i * clipped_frames + j] =
          -processed[(start_token + i) * clipped_frames + j];
    }
  }

  // Run DTW
  DTWResult dtw_result = RunDTW(cost_matrix.data(), effective_tokens, clipped_frames);

#if DTW_DEBUG
  fprintf(stderr, "DTW result: path_length=%zu\n", dtw_result.text_indices.size());
  if (!dtw_result.text_indices.empty()) {
    fprintf(stderr, "  text_indices[0..9]: ");
    for (size_t i = 0; i < std::min(size_t(10), dtw_result.text_indices.size()); ++i) {
      fprintf(stderr, "%d ", dtw_result.text_indices[i]);
    }
    fprintf(stderr, "\n  time_indices[0..9]: ");
    for (size_t i = 0; i < std::min(size_t(10), dtw_result.time_indices.size()); ++i) {
      fprintf(stderr, "%d ", dtw_result.time_indices[i]);
    }
    fprintf(stderr, "\n");
  }
#endif

  // Use "jumps" method to extract times (like OpenAI)
  // jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
  // jump_times = time_indices[jumps] / TOKENS_PER_SECOND
  std::vector<int32_t> token_frames(effective_tokens, 0);

  if (!dtw_result.text_indices.empty()) {
    // Find jumps (where text_idx changes)
    std::vector<int32_t> jump_indices;
    jump_indices.push_back(0);  // First point is always a jump

    for (size_t i = 1; i < dtw_result.text_indices.size(); ++i) {
      if (dtw_result.text_indices[i] != dtw_result.text_indices[i - 1]) {
        jump_indices.push_back(static_cast<int32_t>(i));
      }
    }

    // Extract jump times
    std::vector<int32_t> jump_times;
    for (int32_t idx : jump_indices) {
      if (idx >= 0 && idx < static_cast<int32_t>(dtw_result.time_indices.size())) {
        jump_times.push_back(dtw_result.time_indices[idx]);
      }
    }

#if DTW_DEBUG
    fprintf(stderr, "Jump times (first 10): ");
    for (size_t i = 0; i < std::min(size_t(10), jump_times.size()); ++i) {
      fprintf(stderr, "%.2f ", jump_times[i] * 0.02f);
    }
    fprintf(stderr, "\nTotal jump times: %zu\n", jump_times.size());
#endif

    // Map jump times to tokens
    // Each token's start time is the corresponding jump time
    for (size_t i = 0; i < jump_times.size() && i < static_cast<size_t>(effective_tokens); ++i) {
      token_frames[i] = jump_times[i];
    }
  }

#if DTW_DEBUG
  fprintf(stderr, "========== END DTW DEBUG ==========\n\n");
#endif

  return token_frames;
}

std::vector<float> WhisperDTW::FrameIndicesToSeconds(
    const std::vector<int32_t> &frame_indices) {
  std::vector<float> timestamps;
  timestamps.reserve(frame_indices.size());

  for (int32_t frame : frame_indices) {
    timestamps.push_back(static_cast<float>(frame) * kWhisperSecondsPerToken);
  }

  return timestamps;
}

void WhisperDTW::ProcessAttention(const float *attention, float *output,
                                   int32_t n_heads, int32_t n_tokens,
                                   int32_t n_frames) {
  // Average across heads first
  std::fill(output, output + n_tokens * n_frames, 0.0f);

  for (int32_t h = 0; h < n_heads; ++h) {
    const float *head_data = attention + h * n_tokens * n_frames;
    for (int32_t i = 0; i < n_tokens * n_frames; ++i) {
      output[i] += head_data[i];
    }
  }

  float inv_n_heads = 1.0f / static_cast<float>(n_heads);
  for (int32_t i = 0; i < n_tokens * n_frames; ++i) {
    output[i] *= inv_n_heads;
  }

  // Apply softmax across frames (already done in model, but re-normalize)
  ApplySoftmax(output, n_tokens, n_frames);

  // Apply z-score normalization across tokens
  ApplyZScoreNormalization(output, n_tokens, n_frames);

  // Apply median filter
  ApplyMedianFilter(output, n_tokens, n_frames, 7);
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
