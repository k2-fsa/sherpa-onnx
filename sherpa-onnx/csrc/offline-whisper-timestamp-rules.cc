// sherpa-onnx/csrc/offline-whisper-timestamp-rules.cc
//
// Copyright (c)  2026  Posit Software, PBC

#include "sherpa-onnx/csrc/offline-whisper-timestamp-rules.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace sherpa_onnx {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// =============================================================================
// Step 1: State Determination
// =============================================================================

// Mutually exclusive decoding states
// The expected token pattern is:
//   <|0.00|> text text <|6.60|><|6.60|> text text <|12.00|> EOT
enum class TimestampDecodingState {
  kStart,           // num_sampled == 0: first token must be timestamp
  kAfterOpeningTs,  // last=TS, penult=TS: after opening or double TS, force text
  kSegmentClosing,  // last=TS, penult=text: segment just closed, force TS/EOT
  kInText           // last=text: in text, probability rule may apply
};

// Raw information extracted from the token sequence
struct TokenSequenceInfo {
  int32_t num_sampled;             // tokens sampled so far (excluding SOT sequence)
  bool last_was_timestamp;         // was the last token a timestamp?
  bool penultimate_was_timestamp;  // was the second-to-last token a timestamp?
  int32_t last_ts;                 // last timestamp token ID (-1 if none)
};

// Extract information from the token sequence
TokenSequenceInfo ExtractTokenSequenceInfo(const std::vector<int64_t> &tokens,
                                           int32_t sample_begin,
                                           int32_t timestamp_begin) {
  TokenSequenceInfo info;
  info.num_sampled = static_cast<int32_t>(tokens.size()) - sample_begin;
  info.last_was_timestamp =
      info.num_sampled >= 1 && tokens.back() >= timestamp_begin;
  // IMPORTANT: penultimate defaults to TRUE when len < 2
  // This matches OpenAI's behavior and ensures text follows the first timestamp
  info.penultimate_was_timestamp =
      info.num_sampled < 2 || tokens[tokens.size() - 2] >= timestamp_begin;

  info.last_ts = -1;
  // Find the last timestamp in the sequence (for monotonicity)
  for (int32_t i = sample_begin; i < static_cast<int32_t>(tokens.size()); ++i) {
    if (tokens[i] >= timestamp_begin) {
      info.last_ts = static_cast<int32_t>(tokens[i]);
    }
  }

  return info;
}

// Map raw token info to a mutually exclusive state
TimestampDecodingState DetermineDecodingState(const TokenSequenceInfo &info) {
  if (info.num_sampled == 0) {
    return TimestampDecodingState::kStart;
  }
  if (info.last_was_timestamp && info.penultimate_was_timestamp) {
    return TimestampDecodingState::kAfterOpeningTs;
  }
  if (info.last_was_timestamp && !info.penultimate_was_timestamp) {
    return TimestampDecodingState::kSegmentClosing;
  }
  return TimestampDecodingState::kInText;
}

// =============================================================================
// Step 2: Decision Making
// =============================================================================

// What actions to take based on the current state
struct TimestampDecision {
  bool suppress_text;           // suppress text tokens
  bool suppress_timestamps;     // suppress timestamp tokens
  bool suppress_eot;            // suppress EOT token
  int32_t min_timestamp;        // minimum allowed timestamp (-1 = no constraint)
  int32_t max_timestamp;        // maximum allowed timestamp (-1 = no constraint)
  bool check_probability_rule;  // apply probability rule after other suppressions
};

// Map state to actions - each case must set ALL variables
TimestampDecision DecideTimestampAction(TimestampDecodingState state,
                                        const TokenSequenceInfo &info,
                                        int32_t timestamp_begin,
                                        int32_t max_initial_timestamp_index) {
  // Declare all decision variables - must be set by every case
  bool suppress_text;
  bool suppress_timestamps;
  bool suppress_eot;
  int32_t max_timestamp;
  bool check_probability_rule;

  // Compute monotonicity constraint (cross-cutting concern, used by all cases)
  int32_t min_timestamp = -1;
  if (info.last_ts >= 0) {
    if (state == TimestampDecodingState::kSegmentClosing) {
      // Same timestamp allowed for next segment opening
      min_timestamp = info.last_ts;
    } else {
      // Strictly increasing timestamps
      min_timestamp = info.last_ts + 1;
    }
  }

  switch (state) {
    case TimestampDecodingState::kStart:
      // First token must be a timestamp
      suppress_text = true;
      suppress_timestamps = false;
      suppress_eot = true;
      max_timestamp = (max_initial_timestamp_index >= 0)
                          ? timestamp_begin + max_initial_timestamp_index
                          : -1;
      check_probability_rule = false;
      break;

    case TimestampDecodingState::kAfterOpeningTs:
      // After opening timestamp (or double timestamp), force text
      suppress_text = false;
      suppress_timestamps = true;
      suppress_eot = false;
      max_timestamp = -1;
      check_probability_rule = false;
      break;

    case TimestampDecodingState::kSegmentClosing:
      // Segment just closed, force timestamp or EOT
      suppress_text = true;
      suppress_timestamps = false;
      suppress_eot = false;  // EOT allowed to end transcript
      max_timestamp = -1;
      check_probability_rule = false;
      break;

    case TimestampDecodingState::kInText:
      // In text, probability rule may force timestamp
      suppress_text = false;
      suppress_timestamps = false;
      suppress_eot = false;
      max_timestamp = -1;
      check_probability_rule = true;
      break;
  }

  return TimestampDecision{suppress_text,           suppress_timestamps,
                           suppress_eot,            min_timestamp,
                           max_timestamp,           check_probability_rule};
}

// =============================================================================
// Step 3: Execution
// =============================================================================

// Apply the suppression decisions to the logits
void ApplyTimestampDecision(float *logits, int32_t vocab_size,
                            const TimestampDecision &decision,
                            int32_t timestamp_begin, int32_t eot) {
  // Suppress text tokens if needed
  if (decision.suppress_text) {
    if (decision.suppress_eot) {
      // Suppress all text tokens including EOT
      std::fill(logits, logits + timestamp_begin, kNegInf);
    } else {
      // Suppress text tokens but preserve EOT
      std::fill(logits, logits + eot, kNegInf);
      std::fill(logits + eot + 1, logits + timestamp_begin, kNegInf);
    }
  }

  // Suppress timestamp tokens if needed
  if (decision.suppress_timestamps) {
    std::fill(logits + timestamp_begin, logits + vocab_size, kNegInf);
  }

  // Apply monotonicity constraint (suppress timestamps below minimum)
  if (decision.min_timestamp >= 0) {
    std::fill(logits + timestamp_begin, logits + decision.min_timestamp,
              kNegInf);
  }

  // Apply max_initial constraint (suppress timestamps above maximum)
  if (decision.max_timestamp >= 0) {
    std::fill(logits + decision.max_timestamp + 1, logits + vocab_size,
              kNegInf);
  }
}

// Apply the probability rule: if timestamp probability > max text probability,
// force timestamp. This is the "sum rule" from OpenAI's implementation.
void ApplyProbabilityRule(float *logits, int32_t vocab_size,
                          int32_t timestamp_begin) {
  // Compute logsumexp of timestamp logits
  float max_ts_logit = *std::max_element(logits + timestamp_begin,
                                         logits + vocab_size);
  if (max_ts_logit == kNegInf) {
    return;  // All timestamps suppressed, nothing to do
  }

  float ts_logsum = 0.0f;
  for (int32_t i = timestamp_begin; i < vocab_size; ++i) {
    if (logits[i] > kNegInf) {
      ts_logsum += std::exp(logits[i] - max_ts_logit);
    }
  }
  ts_logsum = max_ts_logit + std::log(ts_logsum);

  // Find max text logit (including EOT - matches OpenAI behavior)
  float max_text_logit = *std::max_element(logits, logits + timestamp_begin);

  // If timestamp logsumexp > max text logit, force timestamp
  if (ts_logsum > max_text_logit) {
    std::fill(logits, logits + timestamp_begin, kNegInf);
  }
}

}  // namespace

// =============================================================================
// Public API
// =============================================================================

void ApplyTimestampRules(float *logits, int32_t vocab_size,
                         const std::vector<int64_t> &tokens,
                         int32_t sample_begin, int32_t timestamp_begin,
                         int32_t no_timestamps, int32_t eot,
                         int32_t max_initial_timestamp_index) {
  // Validate parameters
  assert(logits != nullptr && "logits must not be null");
  assert(vocab_size > 0 && "vocab_size must be positive");
  assert(sample_begin >= 0 && "sample_begin must be non-negative");
  assert(sample_begin <= static_cast<int32_t>(tokens.size()) &&
         "sample_begin must not exceed tokens size");
  assert(timestamp_begin > 0 && "timestamp_begin must be positive");
  assert(timestamp_begin < vocab_size &&
         "timestamp_begin must be less than vocab_size");
  assert(eot >= 0 && eot < timestamp_begin &&
         "eot must be in range [0, timestamp_begin)");
  assert(no_timestamps >= 0 && no_timestamps < vocab_size &&
         "no_timestamps must be in range [0, vocab_size)");

  // Always suppress no_timestamps token
  logits[no_timestamps] = kNegInf;

  // Step 1: Extract token info and determine state
  TokenSequenceInfo info =
      ExtractTokenSequenceInfo(tokens, sample_begin, timestamp_begin);
  TimestampDecodingState state = DetermineDecodingState(info);

  // Step 2: Map state to actions
  TimestampDecision decision =
      DecideTimestampAction(state, info, timestamp_begin,
                            max_initial_timestamp_index);

  // Step 3: Execute the decisions
  ApplyTimestampDecision(logits, vocab_size, decision, timestamp_begin, eot);

  if (decision.check_probability_rule) {
    ApplyProbabilityRule(logits, vocab_size, timestamp_begin);
  }
}

std::vector<OfflineWhisperSegment> ParseTimestampTokens(
    const std::vector<int32_t> &tokens, int32_t timestamp_begin, int32_t eot) {
  // Validate parameters
  assert(timestamp_begin > 0 && "timestamp_begin must be positive");
  assert(eot >= 0 && eot < timestamp_begin &&
         "eot must be in range [0, timestamp_begin)");

  std::vector<OfflineWhisperSegment> segments;

  // Each timestamp token represents 0.02 seconds (20ms)
  constexpr float kSecondsPerTimestamp = 0.02f;

  OfflineWhisperSegment current_segment;
  bool in_segment = false;

  for (size_t i = 0; i < tokens.size(); ++i) {
    int32_t token = tokens[i];

    if (token == eot) {
      // End of transcript - close any open segment
      if (in_segment && !current_segment.token_ids.empty()) {
        current_segment.end_time = -1.0f;  // Use sentinel for EOT-closed segment
        segments.push_back(std::move(current_segment));
        current_segment = OfflineWhisperSegment();
      }
      break;
    }

    if (token >= timestamp_begin) {
      // This is a timestamp token
      float time = (token - timestamp_begin) * kSecondsPerTimestamp;

      if (!in_segment) {
        // Start of a new segment
        current_segment.start_time = time;
        in_segment = true;
      } else {
        // End of current segment
        current_segment.end_time = time;
        if (!current_segment.token_ids.empty()) {
          segments.push_back(std::move(current_segment));
        }
        // Start new segment at same timestamp
        current_segment = OfflineWhisperSegment();
        current_segment.start_time = time;
      }
    } else {
      // Text token - add to current segment
      if (in_segment) {
        current_segment.token_ids.push_back(token);
      }
    }
  }

  // Handle any remaining segment without closing timestamp
  if (in_segment && !current_segment.token_ids.empty()) {
    // Use a sentinel value to indicate incomplete segment
    current_segment.end_time = -1.0f;
    segments.push_back(std::move(current_segment));
  }

  return segments;
}

}  // namespace sherpa_onnx
