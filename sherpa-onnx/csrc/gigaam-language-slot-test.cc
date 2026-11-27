// sherpa-onnx/csrc/gigaam-language-slot-test.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include <cstdint>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/offline-gigaam-model-config.h"

namespace sherpa_onnx {

TEST(GigaAMLanguageSlot, MultilingualV2EncoderOrder) {
  // The multilingual encoder takes a one-hot language vector. Slot 0 means
  // automatic selection, followed by the languages in the model configuration.
  const char *const languages[] = {
      "auto", "ru", "en", "kk", "ky", "uz", "be", "fr", "es", "ba", "ka",
      "hy",   "ro", "tk", "az", "tg", "de", "it", "nl", "pl", "pt", "tt"};
  int32_t slot = 0;
  for (const auto *language : languages) {
    EXPECT_EQ(GigaAMLanguageSlot(language), slot) << language;
    ++slot;
  }
  EXPECT_EQ(GigaAMLanguageSlot("kz"), 3);
  EXPECT_EQ(GigaAMLanguageSlot("xx"), -1);
}

}  // namespace sherpa_onnx
