// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Wed Jun 25 00:22:21 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "bda427f4";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.2";
  return version;
}

}  // namespace sherpa_onnx
