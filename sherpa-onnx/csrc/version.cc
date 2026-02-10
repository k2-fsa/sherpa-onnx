// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Tue Feb 10 11:36:14 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "2ea8c9a0";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.24";
  return version;
}

}  // namespace sherpa_onnx
