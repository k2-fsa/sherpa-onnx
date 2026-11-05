// sherpa-onnx/csrc/nemo-language-prompt.h
//
// Copyright (c)  2026  Julian Pscheid

#ifndef SHERPA_ONNX_CSRC_NEMO_LANGUAGE_PROMPT_H_
#define SHERPA_ONNX_CSRC_NEMO_LANGUAGE_PROMPT_H_

#include <cstdint>
#include <string>
#include <unordered_map>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

// Initialized once from encoder metadata; lookups do not mutate shared state.
class NemoLanguagePrompt {
 public:
  // Unified exports can additionally use the language_prompt_ids metadata key.
  void Init(const Ort::ModelMetadata &meta_data, OrtAllocator *allocator,
            bool use_language_prompt_ids = false);

  // Unified bare-code dictionaries also accept a locale's base language.
  int64_t GetId(const std::string &language,
                bool use_base_language = false) const;

  int64_t DefaultId() const { return default_prompt_id_; }

 private:
  int64_t default_prompt_id_ = 101;
  std::unordered_map<std::string, int64_t> language_prompt_ids_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_NEMO_LANGUAGE_PROMPT_H_
