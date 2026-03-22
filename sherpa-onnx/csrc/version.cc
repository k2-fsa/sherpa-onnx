// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Sun Mar 22 09:06:10 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "4f59df1a";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.32";
  return version;
}

}  // namespace sherpa_onnx
