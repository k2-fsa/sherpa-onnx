// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Sun Apr 12 21:09:56 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "48a3e36c";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.38";
  return version;
}

}  // namespace sherpa_onnx
