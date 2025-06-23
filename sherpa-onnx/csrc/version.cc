// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Fri Jun 20 11:22:52 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "6982b86c";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.1";
  return version;
}

}  // namespace sherpa_onnx
