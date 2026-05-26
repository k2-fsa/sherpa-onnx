// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Wed May 13 18:41:04 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "383b89c4";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.13.2";
  return version;
}

}  // namespace sherpa_onnx
