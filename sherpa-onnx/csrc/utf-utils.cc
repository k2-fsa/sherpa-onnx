// sherpa-onnx/csrc/utf-utils.cc
//
// Copyright      2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/utf-utils.h"

#include <sstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

bool StringToUnicodePoints(const std::string &str,
                           std::vector<int32_t> *codepoints) {
  codepoints->clear();
  const char *data = str.data();
  const size_t length = str.size();
  for (size_t i = 0; i < length; /* no update */) {
    int32_t c = data[i++] & 0xff;
    if ((c & 0x80) == 0) {
      codepoints->push_back(c);
    } else {
      if ((c & 0xc0) == 0x80) {
        return false;
      }
      int32_t count =
          (c >= 0xc0) + (c >= 0xe0) + (c >= 0xf0) + (c >= 0xf8) + (c >= 0xfc);
      int32_t code = c & ((1 << (6 - count)) - 1);
      while (count != 0) {
        if (i == length) {
          return false;
        }
        char cb = data[i++];
        if ((cb & 0xc0) != 0x80) {
          return false;
        }
        code = (code << 6) | (cb & 0x3f);
        count--;
      }
      if (code < 0) {
        // This should not be able to happen.
        return false;
      }
      codepoints->push_back(code);
    }
  }
  return true;
}

std::string CodepointToUTF8String(int32_t code) {
  std::ostringstream ostr;
  if (code < 0) {
    SHERPA_ONNX_LOGE("Invalid code point : $d", code);
    exit(-1);
  } else if (code < 0x80) {
    ostr << static_cast<char>(code);
  } else if (code < 0x800) {
    ostr << static_cast<char>((code >> 6) | 0xc0);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x10000) {
    ostr << static_cast<char>((code >> 12) | 0xe0);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x200000) {
    ostr << static_cast<char>((code >> 18) | 0xf0);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x4000000) {
    ostr << static_cast<char>((code >> 24) | 0xf8);
    ostr << static_cast<char>(((code >> 18) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else {
    ostr << static_cast<char>((code >> 30) | 0xfc);
    ostr << static_cast<char>(((code >> 24) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 18) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  }
  return ostr.str();
}

bool IsCJK(int32_t code) {
  return (code >= 4352 && code <= 4607) || (code >= 11904 && code <= 42191) ||
         (code >= 43072 && code <= 43135) || (code >= 44032 && code <= 55215) ||
         (code >= 63744 && code <= 64255) || (code >= 65072 && code <= 65103) ||
         (code >= 65381 && code <= 65500) || (code >= 131072 && code <= 196607);
}

}  // namespace sherpa_onnx
