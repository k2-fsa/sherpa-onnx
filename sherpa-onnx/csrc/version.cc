// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Thu Jul 10 14:53:33 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "fd9a687e";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.5";
  return version;
}

}  // namespace sherpa_onnx
