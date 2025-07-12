// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Sat Jul 12 19:55:37 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "b376ad3a";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.6";
  return version;
}

}  // namespace sherpa_onnx
