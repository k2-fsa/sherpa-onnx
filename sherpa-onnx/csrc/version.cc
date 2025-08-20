// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Sat Aug 16 18:19:56 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "beb700a9";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.9";
  return version;
}

}  // namespace sherpa_onnx
