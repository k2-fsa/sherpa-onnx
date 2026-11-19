// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

#include "onnxruntime_c_api.h"  // NOLINT

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Thu Nov 19 11:44:29 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "50cc3fc6";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "2.0.2";
  return version;
}

const char *GetOnnxruntimeVersionStr() {
  return OrtGetApiBase()->GetVersionString();
}

}  // namespace sherpa_onnx
