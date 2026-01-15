// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Thu Jan 15 16:00:41 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "80ea3a5f";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.23";
  return version;
}

}  // namespace sherpa_onnx
