// sherpa-onnx/csrc/utf-utils-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/utf-utils.h"

#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(UtfUtils, TestBasic) {
  std::vector<std::string> strs({"Hello world", "你好世界", "Bonjour le monde",
                                 "こんにちは世界", "Привет, мир", "안녕 세계"});
  std::vector<bool> is_cjk({false, true, false, true, false, true});
  std::vector<int32_t> codepoints;
  for (size_t i = 0; i < strs.size(); ++i) {
    std::string str = strs[i];
    StringToUnicodePoints(str, &codepoints);
    EXPECT_EQ(IsCJK(codepoints[0]), is_cjk[i]);
    std::ostringstream oss;
    for (auto code : codepoints) {
      oss << CodepointToUTF8String(code);
    }
    EXPECT_EQ(oss.str(), str);
  }
}
}  // namespace sherpa_onnx
