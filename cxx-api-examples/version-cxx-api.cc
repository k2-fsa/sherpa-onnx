// cxx-api-examples/version-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include <iostream>

#include "sherpa-onnx/c-api/cxx-api.h"

auto main() -> int {
  using namespace sherpa_onnx::cxx;  // NOLINT
  std::cout << "sherpa-onnx version : " << GetVersionStr()
            << "\n";
  std::cout << "sherpa-onnx Git SHA1: " << GetGitSha1() << "\n";
  std::cout << "sherpa-onnx Git date: " << GetGitDate() << "\n";
  std::cout << "onnxruntime version : " << GetOnnxruntimeVersionStr()
            << "\n";

  return 0;
}
