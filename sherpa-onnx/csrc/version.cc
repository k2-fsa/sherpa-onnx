// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Wed Oct 22 12:00:34 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "8a6dec20";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.12.15";
  return version;
}

}  // namespace sherpa_onnx
