// sherpa-onnx/csrc/offline-tts-frontend.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_FRONTEND_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_FRONTEND_H_
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

struct TokenIDs {
  TokenIDs() = default;

  /*implicit*/ TokenIDs(std::vector<int64_t> tokens)  // NOLINT
      : tokens{std::move(tokens)} {}

  TokenIDs(std::vector<int64_t> tokens,  // NOLINT
           std::vector<int64_t> tones)   // NOLINT
      : tokens{std::move(tokens)}, tones{std::move(tones)} {}

  std::string ToString() const;

  std::vector<int64_t> tokens;

  // Used only in MeloTTS
  std::vector<int64_t> tones;
};

class OfflineTtsFrontend {
 public:
  virtual ~OfflineTtsFrontend() = default;

  /** Convert a string to token IDs.
   *
   * @param text The input text.
   *             Example 1: "This is the first sample sentence; this is the
   *             second one." Example 2: "这是第一句。这是第二句。"
   * @param voice Optional. It is for espeak-ng.
   *
   * @return Return a vector-of-vector of token IDs. Each subvector contains
   *         a sentence that can be processed independently.
   *         If a frontend does not support splitting the text into sentences,
   *         the resulting vector contains only one subvector.
   */
  virtual std::vector<TokenIDs> ConvertTextToTokenIds(
      const std::string &text, const std::string &voice = "") const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_FRONTEND_H_
