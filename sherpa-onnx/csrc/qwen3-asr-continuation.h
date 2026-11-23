// sherpa-onnx/csrc/qwen3-asr-continuation.h
//
// Copyright (c) 2026 Qing Peng
#ifndef SHERPA_ONNX_CSRC_QWEN3_ASR_CONTINUATION_H_
#define SHERPA_ONNX_CSRC_QWEN3_ASR_CONTINUATION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {
class QwenAsrTokenizer;

// Internal seam. IDs are the unmodified assistant output, excluding EOS.
struct Qwen3ASRContinuationRequest {
  std::vector<int64_t> prefix_ids;
  bool strict_continuation = false;
};

enum class Qwen3ASRStopReason {
  kEos,
  kNoSpeech,
  kInsufficientFrames,
  kContextLimit,
  kTokenLimit,
  kRepetitionStopped,
  kDecodeError
};

struct Qwen3ASRContinuationResult {
  std::vector<int64_t> raw_output_ids;
  OfflineRecognitionResult result;
  Qwen3ASRStopReason stop_reason = Qwen3ASRStopReason::kDecodeError;
  int64_t prompt_tokens = 0;
  int64_t generated_tokens = 0;
  int64_t context_limit = 0;
  bool had_invalid_utf8 = false;
  double encoder_seconds = 0;
  double decoder_seconds = 0;
  int32_t decoder_calls = 0;
};

std::vector<int64_t> MakeQwen3ASRPrefix(const std::vector<int64_t> &raw_ids,
                                        int32_t chunk_index,
                                        int32_t warmup_chunks,
                                        int32_t rollback_tokens,
                                        QwenAsrTokenizer &tokenizer);

OfflineRecognitionResult FormatQwen3ASRContinuation(
    const std::vector<int64_t> &raw_ids, const std::string &language,
    QwenAsrTokenizer &tokenizer, bool *had_invalid_utf8);

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_QWEN3_ASR_CONTINUATION_H_
