// sherpa-onnx/csrc/offline-whisper-timestamp-rules.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_TIMESTAMP_RULES_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_TIMESTAMP_RULES_H_

#include <cstdint>
#include <vector>

#include "sherpa-onnx/csrc/offline-whisper-decoder.h"

namespace sherpa_onnx {

// Apply OpenAI Whisper's timestamp token rules to logits
// Reference: whisper/decoding.py ApplyTimestampRules
//
// Parameters:
//   logits: pointer to logits array of size vocab_size (modified in-place)
//   vocab_size: size of vocabulary
//   tokens: all tokens decoded so far (including initial SOT sequence)
//   sample_begin: index in tokens where actual sampling began (after SOT seq)
//   timestamp_begin: token ID of first timestamp (<|0.00|>)
//   no_timestamps: token ID of no_timestamps token
//   eot: token ID of end-of-transcript
//   max_initial_timestamp_index: limit for first timestamp (e.g., 50 = 1.0s)
void ApplyTimestampRules(float *logits, int32_t vocab_size,
                         const std::vector<int64_t> &tokens,
                         int32_t sample_begin, int32_t timestamp_begin,
                         int32_t no_timestamps, int32_t eot,
                         int32_t max_initial_timestamp_index);

// Parse timestamp tokens from decoded sequence and create segments
// Pattern: <|start_time|> text tokens... <|end_time|>
//
// Parameters:
//   tokens: decoded tokens (text + timestamp tokens interleaved)
//   timestamp_begin: token ID of first timestamp (<|0.00|>)
//   eot: token ID of end-of-transcript
//
// Returns: vector of segments with start/end times and token IDs
std::vector<OfflineWhisperSegment> ParseTimestampTokens(
    const std::vector<int32_t> &tokens, int32_t timestamp_begin, int32_t eot);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_TIMESTAMP_RULES_H_
