// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Sat Feb 28 15:54:16 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "1934436f";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.28";
  return version;
}

}  // namespace sherpa_onnx
