// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Fri Apr 17 19:03:40 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "c3175218";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.39";
  return version;
}

}  // namespace sherpa_onnx
