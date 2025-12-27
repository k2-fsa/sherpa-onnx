// sherpa-onnx/csrc/offline-whisper-logit-filter.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LOGIT_FILTER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LOGIT_FILTER_H_

#include <cstdint>
#include <vector>

namespace sherpa_onnx {

// This class applies timestamp constraints to Whisper decoder logits.
// Based on the timestamp handling in OpenAI Whisper and whisper.cpp.
//
// The constraints enforce:
// 1. Timestamp pairing: timestamps must come in pairs (start-text-end pattern)
// 2. Non-decreasing: timestamps cannot go backwards in time
// 3. Initial timestamp: first token must be a timestamp
// 4. Probability boosting: force timestamp when aggregate probability is high
class OfflineWhisperLogitFilter {
 public:
  // timestamp_begin: First timestamp token ID (represents 0.00s)
  // timestamp_end: Last timestamp token ID (represents 30.00s)
  // eot_token: End of transcript token ID
  // max_initial_timestamp: Maximum time (in seconds) for the initial timestamp
  OfflineWhisperLogitFilter(int32_t timestamp_begin, int32_t timestamp_end,
                            int32_t eot_token,
                            float max_initial_timestamp = 1.0f);

  // Apply all timestamp constraints to logits.
  // tokens: previously decoded tokens (after the initial prompt)
  // logits: current logits array (modified in-place)
  // vocab_size: size of the vocabulary (length of logits array)
  void Apply(const std::vector<int64_t> &tokens, float *logits,
             int32_t vocab_size) const;

 private:
  // Rule 1: Timestamps must come in pairs
  // If last token was timestamp AND penultimate was also timestamp:
  //   -> Force text token (suppress all timestamps)
  // If last token was timestamp AND penultimate was NOT timestamp:
  //   -> Force another timestamp (suppress all text tokens)
  void ApplyTimestampPairing(const std::vector<int64_t> &tokens, float *logits,
                             int32_t vocab_size) const;

  // Rule 2: Timestamps must be non-decreasing
  // Suppress all timestamp tokens earlier than the last emitted timestamp
  void ApplyNonDecreasing(const std::vector<int64_t> &tokens,
                          float *logits) const;

  // Rule 3: First decoded token must be a timestamp
  // Also limits the initial timestamp to max_initial_timestamp
  void ApplyInitialTimestamp(const std::vector<int64_t> &tokens, float *logits,
                             int32_t vocab_size) const;

  // Rule 4: If sum of timestamp probabilities > max text token probability,
  // force a timestamp (suppress all text tokens)
  void ApplyTimestampProbabilityBoost(const std::vector<int64_t> &tokens,
                                      float *logits, int32_t vocab_size) const;

  int32_t timestamp_begin_;
  int32_t timestamp_end_;
  int32_t eot_token_;
  float max_initial_timestamp_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LOGIT_FILTER_H_
