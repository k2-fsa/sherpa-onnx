// sherpa-onnx/csrc/offline-whisper-lang-probs.h
//
// Copyright (c)  2026  Matias Lin

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LANG_PROBS_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LANG_PROBS_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sherpa_onnx {

// Compute the softmax over logits restricted to the Whisper language token
// IDs in lang_ids. *probs is cleared and filled with {lang_code, prob}
// pairs in the order of lang_ids (first occurrence wins on duplicate IDs).
// Entries whose IDs are missing from id2lang are skipped.
//
// Precondition: every ID in lang_ids must be a valid index into the logits
// buffer. Violating this is undefined behavior.
//
// No-op if either logits or probs is null. If lang_ids is empty or if all
// selected logits are non-finite, *probs is left cleared.
void ComputeWhisperLangProbs(
    const float *logits, const std::vector<int32_t> &lang_ids,
    const std::unordered_map<int32_t, std::string> &id2lang,
    std::vector<std::pair<std::string, float>> *probs);

// Filter *lang_probs in place.
// - top_k == 0: cleared (default: no probs returned).
// - top_k > 0:  keep top N, sorted by prob descending.
// - top_k < 0:  keep all, sorted by prob descending.
void ApplyTopKLangProbs(
    int32_t top_k,
    std::vector<std::pair<std::string, float>> *lang_probs);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LANG_PROBS_H_
