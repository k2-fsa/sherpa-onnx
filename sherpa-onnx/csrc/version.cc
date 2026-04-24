// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Fri Apr 24 16:20:56 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "6be6ddab";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.40";
  return version;
}

}  // namespace sherpa_onnx
