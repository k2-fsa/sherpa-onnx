// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Tue Apr 28 12:10:55 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "47c23919";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.13.0";
  return version;
}

}  // namespace sherpa_onnx
