// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Tue Feb 24 18:42:34 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "66545df5";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.26";
  return version;
}

}  // namespace sherpa_onnx
