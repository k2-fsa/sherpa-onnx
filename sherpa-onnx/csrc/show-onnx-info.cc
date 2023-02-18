// sherpa-onnx/csrc/show-onnx-info.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <iostream>
#include <sstream>

#include "onnxruntime_cxx_api.h"  // NOLINT

int main() {
  std::cout << "ORT_API_VERSION: " << ORT_API_VERSION << "\n";
  std::vector<std::string> providers = Ort::GetAvailableProviders();
  std::ostringstream os;
  os << "Available providers: ";
  std::string sep = "";
  for (const auto &p : providers) {
    os << sep << p;
    sep = ", ";
  }
  std::cout << os.str() << "\n";
  return 0;
}
