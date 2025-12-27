// sherpa-onnx/csrc/offline-whisper-logit-filter.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-logit-filter.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sherpa_onnx {

OfflineWhisperLogitFilter::OfflineWhisperLogitFilter(
    int32_t timestamp_begin, int32_t timestamp_end, int32_t eot_token,
    float max_initial_timestamp)
    : timestamp_begin_(timestamp_begin),
      timestamp_end_(timestamp_end),
      eot_token_(eot_token),
      max_initial_timestamp_(max_initial_timestamp) {}

void OfflineWhisperLogitFilter::Apply(const std::vector<int64_t> &tokens,
                                      float *logits, int32_t vocab_size) const {
  // Apply rules in order
  ApplyInitialTimestamp(tokens, logits, vocab_size);
  ApplyTimestampPairing(tokens, logits, vocab_size);
  ApplyNonDecreasing(tokens, logits);
  ApplyTimestampProbabilityBoost(tokens, logits, vocab_size);
}

void OfflineWhisperLogitFilter::ApplyTimestampPairing(
    const std::vector<int64_t> &tokens, float *logits,
    int32_t vocab_size) const {
  if (tokens.empty()) {
    return;
  }

  int64_t last_token = tokens.back();
  bool last_was_timestamp =
      (last_token >= timestamp_begin_ && last_token <= timestamp_end_);

  if (!last_was_timestamp) {
    return;
  }

  // Check penultimate token
  // Default to true if no penultimate (matching OpenAI Whisper behavior)
  // This allows text after the initial timestamp without forcing pairs
  bool penultimate_was_timestamp = true;
  if (tokens.size() >= 2) {
    int64_t penultimate = tokens[tokens.size() - 2];
    penultimate_was_timestamp =
        (penultimate >= timestamp_begin_ && penultimate <= timestamp_end_);
  }

  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  if (penultimate_was_timestamp) {
    // Two timestamps in a row - force text token (suppress all timestamps)
    // This allows [ts, ts, text, ts, ts] pattern
    for (int32_t i = timestamp_begin_; i < vocab_size; ++i) {
      logits[i] = kNegInf;
    }
  } else {
    // One timestamp after text - force another timestamp (suppress text)
    // This enforces the pairing: text must be bracketed by timestamps
    for (int32_t i = 0; i < eot_token_; ++i) {
      logits[i] = kNegInf;
    }
  }
}

void OfflineWhisperLogitFilter::ApplyNonDecreasing(
    const std::vector<int64_t> &tokens, float *logits) const {
  // Find the last timestamp token in the sequence
  int64_t last_timestamp = -1;
  for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
    if (*it >= timestamp_begin_ && *it <= timestamp_end_) {
      last_timestamp = *it;
      break;
    }
  }

  if (last_timestamp < 0) {
    return;  // No timestamp found
  }

  // Suppress all timestamps that would go backwards in time
  // We allow the same timestamp (for repeated timestamps in pairs)
  // but not earlier ones
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  for (int32_t i = timestamp_begin_; i < last_timestamp; ++i) {
    logits[i] = kNegInf;
  }
}

void OfflineWhisperLogitFilter::ApplyInitialTimestamp(
    const std::vector<int64_t> &tokens, float *logits,
    int32_t vocab_size) const {
  // Check if we've already emitted a timestamp
  bool has_timestamp = false;
  for (auto t : tokens) {
    if (t >= timestamp_begin_ && t <= timestamp_end_) {
      has_timestamp = true;
      break;
    }
  }

  if (has_timestamp) {
    return;  // Already have a timestamp, no need to force initial
  }

  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  // Force initial timestamp - suppress all text tokens
  for (int32_t i = 0; i < timestamp_begin_; ++i) {
    logits[i] = kNegInf;
  }

  // Also limit max initial timestamp
  // Each timestamp token represents 0.02s, so:
  // max_token = timestamp_begin + (max_initial_timestamp / 0.02)
  int32_t max_initial_token =
      timestamp_begin_ + static_cast<int32_t>(max_initial_timestamp_ / 0.02f);

  // Clamp to valid range
  max_initial_token = std::min(max_initial_token, timestamp_end_);

  // Suppress timestamps beyond the max initial
  for (int32_t i = max_initial_token + 1; i <= timestamp_end_; ++i) {
    logits[i] = kNegInf;
  }
}

void OfflineWhisperLogitFilter::ApplyTimestampProbabilityBoost(
    const std::vector<int64_t> &tokens, float *logits,
    int32_t vocab_size) const {
  // This rule: if the sum of probabilities over all timestamp tokens
  // exceeds the maximum probability of any single text token,
  // then force a timestamp.
  //
  // We work in log-space for numerical stability.
  // logsumexp(timestamp_logits) vs max(text_logits)

  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  // Find max logit among text tokens (indices 0 to timestamp_begin-1)
  float max_text_logit = kNegInf;
  for (int32_t i = 0; i < timestamp_begin_; ++i) {
    if (logits[i] > max_text_logit) {
      max_text_logit = logits[i];
    }
  }

  // Compute logsumexp of timestamp logits
  // First find max for numerical stability
  float max_ts_logit = kNegInf;
  for (int32_t i = timestamp_begin_; i <= timestamp_end_; ++i) {
    if (logits[i] > max_ts_logit) {
      max_ts_logit = logits[i];
    }
  }

  if (max_ts_logit == kNegInf) {
    return;  // All timestamps suppressed, nothing to do
  }

  // logsumexp = max + log(sum(exp(x - max)))
  float sum_exp = 0.0f;
  for (int32_t i = timestamp_begin_; i <= timestamp_end_; ++i) {
    if (logits[i] > kNegInf) {
      sum_exp += std::exp(logits[i] - max_ts_logit);
    }
  }

  float timestamp_logsumexp = max_ts_logit + std::log(sum_exp);

  // If aggregate timestamp probability > max text probability, force timestamp
  if (timestamp_logsumexp > max_text_logit) {
    for (int32_t i = 0; i < timestamp_begin_; ++i) {
      logits[i] = kNegInf;
    }
  }
}

}  // namespace sherpa_onnx
