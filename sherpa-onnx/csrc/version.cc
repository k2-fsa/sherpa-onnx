// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Thu Feb 26 18:07:07 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "ad53853f";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.27";
  return version;
}

}  // namespace sherpa_onnx
