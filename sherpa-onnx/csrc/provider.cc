// sherpa-onnx/csrc/provider.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/provider.h"

#include <algorithm>
#include <cctype>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

Provider StringToProvider(std::string s) {
  std::transform(s.cbegin(), s.cend(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (s == "cpu") {
    return Provider::kCPU;
  } else if (s == "cuda") {
    return Provider::kCUDA;
  } else if (s == "coreml") {
    return Provider::kCoreML;
  } else {
    SHERPA_ONNX_LOGE("Unsupported string: %s. Fallback to cpu", s.c_str());
    return Provider::kCPU;
  }
}

}  // namespace sherpa_onnx
