// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Thu Mar 26 19:14:29 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "766a15c0";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.34";
  return version;
}

}  // namespace sherpa_onnx
