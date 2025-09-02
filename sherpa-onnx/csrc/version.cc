// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Mon Sep 1 11:59:46 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "f0e68cde";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.11";
  return version;
}

}  // namespace sherpa_onnx
