// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Fri May 8 22:02:28 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "2a3a9d95";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.13.1";
  return version;
}

}  // namespace sherpa_onnx
