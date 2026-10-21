// sherpa-onnx/csrc/unicode-case-mapping.h
//
// GENERATED FILE - DO NOT EDIT BY HAND.
//
// Regenerate with scripts/gen-unicode-case-mapping.py (which reads Python's
// `unicodedata`). These tables replace the hand-maintained switch in
// ToLowerCase() that could only cover a few dozen characters and silently
// left the rest (e.g. U+0100 LATIN CAPITAL LETTER A WITH MACRON) uppercase
// because std::towlower() is a no-op for non-ASCII in the "C" locale.
//
// This file is part of sherpa-onnx.
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_UNICODE_CASE_MAPPING_H_
#define SHERPA_ONNX_CSRC_UNICODE_CASE_MAPPING_H_

#include <cstdint>
#include <string>

namespace sherpa_onnx {

// Simple (1:1) Unicode lowercase mapping. Returns cp unchanged when there is
// no single-codepoint lowercase form. Not locale-dependent, unlike
// std::towlower().
char32_t ToSimpleLower(char32_t cp);

// True if cp is a Unicode Letter (category L*) or Mark (category M*), i.e.
// something that can appear inside a lexicon word. Used to strip/split
// punctuation away from OOV tokens before retrying the lexicon.
bool IsLexiconWordChar(char32_t cp);

// Canonical combining class of |cp| (0 for starters).
uint32_t GetCanonicalCombiningClass(char32_t cp);

// In-place Unicode NFC (canonical decomposition, canonical ordering,
// canonical composition). Needed so that e.g. "a" + U+0301 matches a lexicon
// entry stored as U+00E1. Returns true if |s| changed.
bool NormalizeNfc(std::u32string *s);

// Convenience overload for UTF-8 input.
std::string NormalizeNfc(const std::string &s);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_UNICODE_CASE_MAPPING_H_
