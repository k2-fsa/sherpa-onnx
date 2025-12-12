// sherpa-onnx/csrc/vec-to-string.h
//
// Utility to convert std::vector<T> to a string.
//
// Copyright (c) 2025

#ifndef SHERPA_ONNX_CSRC_VEC_TO_STRING_H_
#define SHERPA_ONNX_CSRC_VEC_TO_STRING_H_

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace sherpa_onnx {

// Format:
//   numeric: [1, 2, 3]
//   string : ["a", "b"]
template <typename T>
inline std::string VecToString(const std::vector<T> &vec,
                               int32_t precision = 6) {
  std::ostringstream oss;
  if (precision != 0) {
    oss << std::fixed << std::setprecision(precision);
  }
  oss << "[";
  std::string sep = "";
  for (const auto &item : vec) {
    oss << sep << item;
    sep = ", ";
  }
  oss << "]";
  return oss.str();
}

// Explicit specialization for std::string
template <>
inline std::string VecToString<std::string>(const std::vector<std::string> &vec,
                                            int32_t /*precision*/) {
  std::ostringstream oss;
  oss << "[";
  std::string sep = "";
  for (const auto &item : vec) {
    oss << sep << std::quoted(item);
    sep = ", ";
  }
  oss << "]";
  return oss.str();
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VEC_TO_STRING_H_