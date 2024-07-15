// sherpa-onnx/csrc/utfcpp-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include <cctype>
#include <string>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

TEST(UTF8, Case1) {
  std::string hello = "你好, 早上好！世界.  hello!。Hallo! how are you?";
  std::vector<std::string> ss = SplitUtf8(hello);
  for (const auto &s : ss) {
    std::cout << s << "\n";
  }
}

}  // namespace sherpa_onnx
