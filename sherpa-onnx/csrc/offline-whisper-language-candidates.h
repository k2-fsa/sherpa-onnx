// sherpa-onnx/csrc/offline-whisper-language-candidates.h
//
// Copyright (c)  2026  Ngoroth

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LANGUAGE_CANDIDATES_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LANGUAGE_CANDIDATES_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

// Call only when the stream has language_candidates. An absent option is
// represented separately by an empty ID vector, meaning unrestricted detection.
// On failure, ids is empty and must not be used to fall back to unrestricted.
inline bool ResolveWhisperLanguageCandidates(
    const std::string &csv, const std::string &language, bool is_multilingual,
    const std::unordered_map<std::string, int32_t> &lang2id,
    std::vector<int32_t> *ids) {
  ids->clear();
  if (!language.empty()) {
    SHERPA_ONNX_LOGE(
        "language_candidates requires automatic language detection; "
        "the configured language is '%s'",
        language.c_str());
    return false;
  }

  if (!is_multilingual) {
    SHERPA_ONNX_LOGE("language_candidates requires a multilingual Whisper model");
    return false;
  }

  std::vector<std::string> codes;
  SplitStringToVector(csv, ",", false, &codes);
  std::vector<int32_t> resolved;
  resolved.reserve(codes.size());
  for (std::size_t i = 0; i != codes.size(); ++i) {
    codes[i] = Trim(codes[i]);
    const auto &code = codes[i];
    if (code.empty()) {
      SHERPA_ONNX_LOGE("language_candidates contains an empty language code");
      return false;
    }

    if (std::find(codes.begin(), codes.begin() + i, code) != codes.begin() + i) {
      SHERPA_ONNX_LOGE("Duplicate language_candidates code: '%s'", code.c_str());
      return false;
    }

    auto it = lang2id.find(code);
    if (it == lang2id.end()) {
      SHERPA_ONNX_LOGE(
          "Unknown language_candidates code in Whisper model metadata: '%s'",
          code.c_str());
      return false;
    }
    resolved.push_back(it->second);
  }

  *ids = std::move(resolved);
  return true;
}

// ids must be nonempty and contain valid metadata token IDs. Exact ties are
// resolved by the order of ids, for both restricted and unrestricted detection.
inline int32_t SelectWhisperLanguage(const float *logits,
                                     const std::vector<int32_t> &ids) {
  int32_t best = ids[0];
  float best_logit = logits[best];
  for (std::size_t i = 1; i != ids.size(); ++i) {
    int32_t id = ids[i];
    if (logits[id] > best_logit) {
      best = id;
      best_logit = logits[id];
    }
  }
  return best;
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_LANGUAGE_CANDIDATES_H_
