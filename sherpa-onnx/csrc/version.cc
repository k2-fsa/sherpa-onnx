// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Tue Mar 24 15:30:03 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "b1f94831";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.33";
  return version;
}

}  // namespace sherpa_onnx
