// sherpa-onnx/csrc/offline-whisper-lang-probs.h
//
// Copyright (c)  2026  Matias Lin

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LANG_PROBS_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LANG_PROBS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

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
inline void ComputeWhisperLangProbs(
    const float *logits, const std::vector<int32_t> &lang_ids,
    const std::unordered_map<int32_t, std::string> &id2lang,
    std::vector<std::pair<std::string, float>> *probs) {
  if (logits == nullptr || probs == nullptr) {
    return;
  }

  probs->clear();

  if (lang_ids.empty()) return;

  // Dedup lang_ids (first occurrence wins) to avoid overweighting duplicated
  // token IDs in the softmax.
  std::vector<int32_t> unique_ids;
  unique_ids.reserve(lang_ids.size());
  std::unordered_set<int32_t> seen;
  seen.reserve(lang_ids.size());
  for (int32_t id : lang_ids) {
    if (seen.insert(id).second) unique_ids.push_back(id);
  }

  float max_logit = logits[unique_ids[0]];
  for (std::size_t i = 1; i != unique_ids.size(); ++i) {
    float l = logits[unique_ids[i]];
    if (l > max_logit) max_logit = l;
  }
  // All logits are non-finite.
  if (!std::isfinite(max_logit)) return;

  std::vector<float> exps;
  exps.reserve(unique_ids.size());
  double sum = 0.0;
  for (int32_t id : unique_ids) {
    float e = std::exp(logits[id] - max_logit);
    exps.push_back(e);
    sum += e;
  }

  float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
  probs->reserve(unique_ids.size());
  for (std::size_t i = 0; i != unique_ids.size(); ++i) {
    auto it = id2lang.find(unique_ids[i]);
    if (it == id2lang.end()) continue;
    probs->emplace_back(it->second, exps[i] * inv_sum);
  }
}

// Filter *lang_probs in place based on the per-stream "top_k_langs" option:
// - ""/"0" or unparseable: cleared (default: no probs returned).
// - "N" > 0: keep top N (int), sorted by prob descending.
// - "-1": keep all, sorted by prob descending.
inline void ApplyTopKLangProbs(
    const std::string &top_k_langs,
    std::vector<std::pair<std::string, float>> *lang_probs) {
  if (lang_probs == nullptr) return;

  if (top_k_langs.empty()) {
    lang_probs->clear();
    return;
  }

  int32_t top_k = 0;
  try {
    top_k = std::stoi(top_k_langs);
  } catch (const std::exception &ex) {
    SHERPA_ONNX_LOGE(
        "Failed to parse 'top_k_langs' value '%s' as int: %s. "
        "Falling back to default (no lang probs returned).",
        top_k_langs.c_str(), ex.what());
    lang_probs->clear();
    return;
  }

  if (top_k == 0) {
    lang_probs->clear();
    return;
  }

  std::sort(lang_probs->begin(), lang_probs->end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  if (top_k > 0 && static_cast<int32_t>(lang_probs->size()) > top_k) {
    lang_probs->resize(top_k);
  }
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LANG_PROBS_H_
