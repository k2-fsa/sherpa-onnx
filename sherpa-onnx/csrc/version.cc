// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Sat Feb 14 22:31:54 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "20429103";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.25";
  return version;
}

}  // namespace sherpa_onnx
