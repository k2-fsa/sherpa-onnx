// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Mon Jul 28 00:58:27 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "2bc632d6";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.7";
  return version;
}

}  // namespace sherpa_onnx
