// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Sat Apr 11 00:38:25 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "932f9164";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.37";
  return version;
}

}  // namespace sherpa_onnx
