// sherpa-onnx/csrc/nemo-language-prompt.cc
//
// Copyright (c)  2026  Julian Pscheid

#include "sherpa-onnx/csrc/nemo-language-prompt.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

std::string StripQuotes(std::string s) {
  s = Trim(s);
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                        (s.front() == '\'' && s.back() == '\''))) {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

std::string NormalizeLanguage(std::string s) {
  s = StripQuotes(std::move(s));
  s = Trim(s);
  if (s.size() >= 2 && s.front() == '<' && s.back() == '>') {
    s = s.substr(1, s.size() - 2);
  }

  std::replace(s.begin(), s.end(), '_', '-');
  ToLowerCase(&s);
  return s;
}

void AddLanguagePromptId(
    const std::string &language, int64_t prompt_id,
    std::unordered_map<std::string, int64_t> *language_prompt_ids,
    std::vector<std::pair<std::string, int64_t>> *ordered_prompt_ids) {
  auto normalized = NormalizeLanguage(language);
  if (normalized.empty()) {
    return;
  }

  if (language_prompt_ids->emplace(normalized, prompt_id).second) {
    ordered_prompt_ids->emplace_back(normalized, prompt_id);
  }
}

void AddBaseLanguageAliases(
    std::unordered_map<std::string, int64_t> *language_prompt_ids,
    const std::vector<std::pair<std::string, int64_t>> &ordered_prompt_ids) {
  for (const auto &p : ordered_prompt_ids) {
    auto pos = p.first.find('-');
    if (pos == std::string::npos || pos == 0) {
      continue;
    }

    auto base = p.first.substr(0, pos);
    language_prompt_ids->emplace(std::move(base), p.second);
  }
}

// The metadata value is a JSON object mapping language strings to integer
// prompt ids, e.g. {"auto": 101, "en-US": 0, "ja-JP": 1}.
void ParseLanguagePromptDictionary(
    const std::string &value,
    std::unordered_map<std::string, int64_t> *language_prompt_ids,
    std::vector<std::pair<std::string, int64_t>> *ordered_prompt_ids) {
  auto j = nlohmann::json::parse(value, nullptr, /*allow_exceptions*/ false);
  if (j.is_discarded() || !j.is_object()) {
    return;
  }

  for (const auto &item : j.items()) {
    if (!item.value().is_number_integer()) {
      continue;
    }

    int64_t prompt_id = item.value().get<int64_t>();
    if (prompt_id < 0) {
      continue;
    }

    AddLanguagePromptId(item.key(), prompt_id, language_prompt_ids,
                        ordered_prompt_ids);
  }
}

bool ReadPromptIdFromMetadata(const Ort::ModelMetadata &meta_data,
                              OrtAllocator *allocator, const char *key,
                              int64_t *prompt_id) {
  auto value = LookupCustomModelMetaData(meta_data, key, allocator);
  if (value.empty()) {
    return false;
  }

  int64_t id = -1;
  if (!ConvertStringToInteger(Trim(value), &id) || id < 0) {
    return false;
  }

  *prompt_id = id;
  return true;
}

}  // namespace

void NemoLanguagePrompt::Init(const Ort::ModelMetadata &meta_data,
                              OrtAllocator *allocator,
                              bool use_language_prompt_ids) {
  std::vector<std::pair<std::string, int64_t>> ordered_prompt_ids;
  auto dict =
      LookupCustomModelMetaData(meta_data, "prompt_dictionary", allocator);
  if (!dict.empty()) {
    ParseLanguagePromptDictionary(dict, &language_prompt_ids_,
                                  &ordered_prompt_ids);
  }

  if (use_language_prompt_ids) {
    auto alias =
        LookupCustomModelMetaData(meta_data, "language_prompt_ids", allocator);
    if (!alias.empty()) {
      ParseLanguagePromptDictionary(alias, &language_prompt_ids_,
                                    &ordered_prompt_ids);
    }
  }

  ReadPromptIdFromMetadata(meta_data, allocator, "auto_prompt_id",
                           &default_prompt_id_);

  auto it = language_prompt_ids_.find("auto");
  if (it != language_prompt_ids_.end()) {
    default_prompt_id_ = it->second;
  } else {
    AddLanguagePromptId("auto", default_prompt_id_, &language_prompt_ids_,
                        &ordered_prompt_ids);
  }

  bool has_non_auto_prompt = false;
  for (const auto &p : ordered_prompt_ids) {
    if (p.first != "auto") {
      has_non_auto_prompt = true;
      break;
    }
  }
  if (!has_non_auto_prompt) {
    SHERPA_ONNX_LOGE(
        "The encoder declares prompt_index, but usable prompt_dictionary "
        "metadata is missing; all languages will fall back to auto.");
  }

  AddBaseLanguageAliases(&language_prompt_ids_, ordered_prompt_ids);
}

int64_t NemoLanguagePrompt::GetId(const std::string &language,
                                  bool use_base_language) const {
  if (language.empty()) {
    return default_prompt_id_;
  }

  auto it = language_prompt_ids_.find(language);
  if (it != language_prompt_ids_.end()) {
    return it->second;
  }

  auto normalized = NormalizeLanguage(language);
  if (normalized.empty() || normalized == "auto") {
    return default_prompt_id_;
  }

  it = language_prompt_ids_.find(normalized);
  if (it != language_prompt_ids_.end()) {
    return it->second;
  }

  if (use_base_language) {
    auto pos = normalized.find('-');
    if (pos != std::string::npos) {
      it = language_prompt_ids_.find(normalized.substr(0, pos));
      if (it != language_prompt_ids_.end()) {
        return it->second;
      }
    }
  }

  SHERPA_ONNX_LOGE(
      "Unsupported language '%s' for multilingual NeMo transducer; using "
      "auto",
      language.c_str());
  return default_prompt_id_;
}

}  // namespace sherpa_onnx
