// sherpa-onnx/csrc/text-utils-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/text-utils.h"

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(RemoveInvalidUtf8Sequences, Case1) {
  std::vector<uint8_t> v = {
      0xe4, 0xbb, 0x8a,                                  // 今
      0xe5, 0xa4, 0xa9,                                  // 天
      'i',  's',  ' ',  'M', 'o', 'd', 'a', 'y',  ',',   // is Monday,
      ' ',  'w',  'i',  'e', ' ', 'h', 'e', 'i',  0xc3,  // wie heißen Size
      0x9f, 'e',  'n',  ' ', 'S', 'i', 'e', 0xf0, 0x9d, 0x84, 0x81};

  std::vector<uint8_t> v0 = v;
  v0[1] = 0xc0;  // make the first 3 bytes an invalid utf8 character
  std::string s0{v0.begin(), v0.end()};
  EXPECT_EQ(s0.size(), v0.size());

  auto s = RemoveInvalidUtf8Sequences(s0);  // should remove 今

  v0 = v;
  // v0[23] == 0xc3
  // v0[24] == 0x9f

  v0[23] = 0xc1;

  s0 = {v0.begin(), v0.end()};
  s = RemoveInvalidUtf8Sequences(s0);  // should remove ß

  EXPECT_EQ(s.size() + 2, v.size());

  v0 = v;
  // v0[31] = 0xf0;
  // v0[32] = 0x9d;
  // v0[33] = 0x84;
  // v0[34] = 0x81;
  v0[31] = 0xf5;

  s0 = {v0.begin(), v0.end()};
  s = RemoveInvalidUtf8Sequences(s0);

  EXPECT_EQ(s.size() + 4, v.size());
}

}  // namespace sherpa_onnx
