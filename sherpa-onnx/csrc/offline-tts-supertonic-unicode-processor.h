// sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor.h
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_UNICODE_PROCESSOR_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_UNICODE_PROCESSOR_H_

#include <string>
#include <vector>

namespace sherpa_onnx {

// Available languages for multilingual TTS
extern const std::vector<std::string> kSupertonicAvailableLangs;

// Unicode text processor for Supertonic TTS
class SupertonicUnicodeProcessor {
 public:
  explicit SupertonicUnicodeProcessor(const std::string &unicode_indexer_path);

  template <typename Manager>
  SupertonicUnicodeProcessor(Manager *mgr,
                             const std::string &unicode_indexer_path);

  // Process text list to text IDs and mask
  void Process(const std::vector<std::string> &text_list,
               const std::vector<std::string> &lang_list,
               std::vector<std::vector<int64_t>> *text_ids,
               std::vector<std::vector<std::vector<float>>> *text_mask) const;

 private:
  std::string PreprocessText(const std::string &text,
                             const std::string &lang) const;
  std::vector<uint16_t> TextToUnicodeValues(const std::string &text) const;
  std::vector<std::vector<std::vector<float>>> GetTextMask(
      const std::vector<int64_t> &text_ids_lengths) const;

  std::vector<int64_t> indexer_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_UNICODE_PROCESSOR_H_
