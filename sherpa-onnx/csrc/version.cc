// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Wed Dec 17 19:57:55 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "3290e1ce";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.20";
  return version;
}

}  // namespace sherpa_onnx
