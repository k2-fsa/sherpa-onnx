// sherpa-onnx/csrc/offline-whisper-timestamp-rules.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-timestamp-rules.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace sherpa_onnx {

void ApplyTimestampRules(float *logits, int32_t vocab_size,
                         const std::vector<int64_t> &tokens,
                         int32_t sample_begin, int32_t timestamp_begin,
                         int32_t no_timestamps, int32_t eot,
                         int32_t max_initial_timestamp_index) {
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  // 1. Always suppress no_timestamps token
  logits[no_timestamps] = kNegInf;

  // Number of tokens sampled so far (excluding initial SOT sequence)
  int32_t num_sampled = static_cast<int32_t>(tokens.size()) - sample_begin;

  // 2. Determine state: was last token a timestamp? penultimate?
  // IMPORTANT: penultimate_was_timestamp defaults to TRUE when len < 2
  // This matches OpenAI's behavior and ensures text follows the first timestamp
  bool last_was_timestamp = false;
  bool penultimate_was_timestamp = (num_sampled < 2);  // Default true if not enough tokens
  int32_t last_timestamp_id = -1;

  if (num_sampled >= 1) {
    int64_t last_token = tokens.back();
    last_was_timestamp = (last_token >= timestamp_begin);
    if (last_was_timestamp) {
      last_timestamp_id = static_cast<int32_t>(last_token);
    }
  }
  if (num_sampled >= 2) {
    int64_t penult_token = tokens[tokens.size() - 2];
    penultimate_was_timestamp = (penult_token >= timestamp_begin);
  }

  // 3. Timestamp pairing rules (matches OpenAI's logic)
  // Pattern: <|0.00|> text text <|6.60|><|6.60|> text text <|12.00|> EOT
  //
  // - After first timestamp: penultimate defaults to true, so text is forced
  // - After text then timestamp: penultimate=false, so next must be timestamp/EOT
  // - After two timestamps: penultimate=true, so text is forced
  if (last_was_timestamp) {
    if (penultimate_was_timestamp) {
      // After two consecutive timestamps (or after first timestamp), force text
      std::fill(logits + timestamp_begin, logits + vocab_size, kNegInf);
    } else {
      // After single timestamp following text, suppress text tokens
      // Only allow timestamps (for next segment) or EOT
      std::fill(logits, logits + eot, kNegInf);
      // EOT is allowed (index eot is not suppressed)
      std::fill(logits + eot + 1, logits + timestamp_begin, kNegInf);
    }
  }

  // 4. Monotonicity: timestamps must not decrease
  // Find the last timestamp in the sequence
  int32_t last_ts = -1;
  for (int32_t i = sample_begin; i < static_cast<int32_t>(tokens.size()); ++i) {
    if (tokens[i] >= timestamp_begin) {
      last_ts = static_cast<int32_t>(tokens[i]);
    }
  }
  if (last_ts >= 0) {
    // Suppress timestamps before the last one.
    // Also force each segment to have nonzero length to prevent infinite looping:
    // - After text+timestamp (segment just closed): allow same timestamp
    //   because it could be the opening timestamp of the next segment
    // - Otherwise: force strictly increasing (use last_ts + 1)
    int32_t suppress_up_to;
    if (last_was_timestamp && !penultimate_was_timestamp) {
      // Segment just closed - next segment can start at the same time
      suppress_up_to = last_ts;
    } else {
      // Force strictly increasing timestamps
      suppress_up_to = last_ts + 1;
    }
    std::fill(logits + timestamp_begin, logits + suppress_up_to, kNegInf);
  }

  // 5. First sampled token must be a timestamp
  if (num_sampled == 0) {
    // Suppress all text tokens
    std::fill(logits, logits + timestamp_begin, kNegInf);
    // Apply max_initial_timestamp constraint
    if (max_initial_timestamp_index >= 0) {
      int32_t max_ts = timestamp_begin + max_initial_timestamp_index;
      std::fill(logits + max_ts + 1, logits + vocab_size, kNegInf);
    }
  }

  // 6. Probability rule: if timestamp probability > max text probability,
  //    force timestamp. This is the "sum rule" from OpenAI's implementation.
  if (!last_was_timestamp && num_sampled > 0) {
    // Compute logsumexp of timestamp logits
    float max_ts_logit = kNegInf;
    for (int32_t i = timestamp_begin; i < vocab_size; ++i) {
      if (logits[i] > max_ts_logit) {
        max_ts_logit = logits[i];
      }
    }

    float ts_logsum = 0.0f;
    if (max_ts_logit > kNegInf) {
      for (int32_t i = timestamp_begin; i < vocab_size; ++i) {
        if (logits[i] > kNegInf) {
          ts_logsum += std::exp(logits[i] - max_ts_logit);
        }
      }
      ts_logsum = max_ts_logit + std::log(ts_logsum);
    } else {
      ts_logsum = kNegInf;
    }

    // Find max text logit (including EOT - matches OpenAI behavior)
    float max_text_logit = *std::max_element(logits, logits + timestamp_begin);

    // If timestamp logsumexp > max text logit, force timestamp
    if (ts_logsum > max_text_logit) {
      std::fill(logits, logits + timestamp_begin, kNegInf);
    }
  }
}

std::vector<OfflineWhisperSegment> ParseTimestampTokens(
    const std::vector<int32_t> &tokens, int32_t timestamp_begin, int32_t eot) {
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
