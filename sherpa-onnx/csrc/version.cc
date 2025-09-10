// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Wed Sep 10 18:52:18 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "7e42ba2c";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.12";
  return version;
}

}  // namespace sherpa_onnx
