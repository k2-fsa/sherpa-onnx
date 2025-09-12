// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Fri Sep 12 15:54:03 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "c415092f";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.13";
  return version;
}

}  // namespace sherpa_onnx
