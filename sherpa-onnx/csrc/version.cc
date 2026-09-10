// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

#include "onnxruntime_c_api.h"  // NOLINT

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Thu Sep 10 17:15:24 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "8c8e275d";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "1.13.8";
  return version;
}

const char *GetOnnxruntimeVersionStr() {
  return OrtGetApiBase()->GetVersionString();
}

}  // namespace sherpa_onnx
