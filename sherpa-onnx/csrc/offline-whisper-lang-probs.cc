// sherpa-onnx/csrc/offline-whisper-lang-probs.cc
//
// Copyright (c)  2026  Matias Lin

#include "sherpa-onnx/csrc/offline-whisper-lang-probs.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {

void ComputeWhisperLangProbs(
    const float *logits, const std::vector<int32_t> &lang_ids,
    const std::unordered_map<int32_t, std::string> &id2lang,
    std::vector<std::pair<std::string, float>> *probs) {
  if (logits == nullptr || probs == nullptr) {
    return;
  }

  probs->clear();

  if (lang_ids.empty()) return;

  // Dedup lang_ids to avoid overweighting duplicated token IDs in the softmax.
  std::vector<int32_t> unique_ids = Deduplicate(lang_ids);
  std::vector<float> values;
  values.reserve(unique_ids.size());
  for (int32_t id : unique_ids) values.push_back(logits[id]);

  // Guard against all logits being non-finite.
  if (!std::isfinite(*std::max_element(values.begin(), values.end()))) return;

  bool rc = Softmax(values.data(), static_cast<int32_t>(values.size()));
  if (!rc) return;

  probs->reserve(unique_ids.size());
  for (std::size_t i = 0; i != unique_ids.size(); ++i) {
    auto it = id2lang.find(unique_ids[i]);
    if (it == id2lang.end()) continue;
    probs->emplace_back(it->second, values[i]);
  }
}

void ApplyTopKLangProbs(
    int32_t top_k,
    std::vector<std::pair<std::string, float>> *lang_probs) {
  if (lang_probs == nullptr) return;

  if (top_k == 0) {
    lang_probs->clear();
    return;
  }

  int32_t size = static_cast<int32_t>(lang_probs->size());
  // Negative means keep all.
  int32_t k = (top_k < 0) ? size : top_k;

  std::vector<float> probs_only;
  probs_only.reserve(size);
  for (const auto &lp : *lang_probs) {
    probs_only.push_back(lp.second);
  }

  std::vector<int32_t> top_indices = TopkIndex(probs_only.data(), size, k);

  std::vector<std::pair<std::string, float>> out;
  out.reserve(top_indices.size());
  for (int32_t idx : top_indices) {
    out.push_back((*lang_probs)[idx]);
  }
  *lang_probs = std::move(out);
}

}  // namespace sherpa_onnx
