// sherpa-onnx/csrc/offline-tts-frontend.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-frontend.h"

#include <sstream>
#include <string>

namespace sherpa_onnx {

std::string TokenIDs::ToString() const {
  std::ostringstream os;
  os << "TokenIDs(";
  os << "tokens=[";
  std::string sep;
  for (auto i : tokens) {
    os << sep << i;
    sep = ", ";
  }
  os << "], ";

  os << "tones=[";
  sep = {};
  for (auto i : tones) {
    os << sep << i;
    sep = ", ";
  }
  os << "]";
  os << ")";
  return os.str();
}

}  // namespace sherpa_onnx
