// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Fri Jul 4 15:57:07 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "3bf986d0";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.4";
  return version;
}

}  // namespace sherpa_onnx
