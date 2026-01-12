// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Mon Jan 12 19:05:54 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "82b17d2f";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.21";
  return version;
}

}  // namespace sherpa_onnx
