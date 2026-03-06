// sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor.h
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_UNICODE_PROCESSOR_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_UNICODE_PROCESSOR_H_

#include <cstdint>
#include <string>
#include <vector>

namespace sherpa_onnx {

// Unicode text processor for Supertonic TTS
class SupertonicUnicodeProcessor {
 public:
  explicit SupertonicUnicodeProcessor(const std::string &unicode_indexer_path);

  template <typename Manager>
  SupertonicUnicodeProcessor(Manager *mgr,
                             const std::string &unicode_indexer_path);

  void Process(const std::string &text, const std::string &lang,
               std::vector<int64_t> *text_ids,
               std::vector<float> *text_mask_flat,
               std::vector<int64_t> *text_mask_shape) const;

 private:
  std::string PreprocessText(const std::string &text,
                             const std::string &lang) const;
  std::vector<uint16_t> TextToUnicodeValues(const std::string &text) const;

  std::vector<int32_t> indexer_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_UNICODE_PROCESSOR_H_
