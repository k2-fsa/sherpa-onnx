// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Tue Jul 7 19:00:41 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "753609d0";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.13.4";
  return version;
}

}  // namespace sherpa_onnx
