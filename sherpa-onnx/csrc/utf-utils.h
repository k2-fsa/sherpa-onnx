// sherpa-onnx/csrc/utf-utils.h
//
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_UTF_UTILS_H_
#define SHERPA_ONNX_CSRC_UTF_UTILS_H_

#include <string>
#include <vector>

namespace sherpa_onnx {

bool StringToUnicodePoints(const std::string &str,
                           std::vector<int32_t> *codepoints);

std::string CodepointToUTF8String(int32_t codepoint);

bool IsCJK(int32_t codepoint);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_UTF_UTILS_H_
