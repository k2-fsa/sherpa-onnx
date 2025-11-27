// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Thu Nov 27 15:24:39 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "7d1d2270";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.18";
  return version;
}

}  // namespace sherpa_onnx
