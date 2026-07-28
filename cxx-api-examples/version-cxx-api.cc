// cxx-api-examples/version-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include <iostream>

#include "sherpa-onnx/c-api/cxx-api.h"

auto main() -> int {
  std::cout << "sherpa-onnx version : " << sherpa_onnx::cxx::GetVersionStr()
            << "\n";
  std::cout << "sherpa-onnx Git SHA1: " << sherpa_onnx::cxx::GetGitSha1()
            << "\n";
  std::cout << "sherpa-onnx Git date: " << sherpa_onnx::cxx::GetGitDate()
            << "\n";
  std::cout << "onnxruntime version : "
            << sherpa_onnx::cxx::GetOnnxruntimeVersionStr() << "\n";

  return 0;
}
