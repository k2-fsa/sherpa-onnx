// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Fri Aug 8 20:26:15 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "17c735da";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.8";
  return version;
}

}  // namespace sherpa_onnx
