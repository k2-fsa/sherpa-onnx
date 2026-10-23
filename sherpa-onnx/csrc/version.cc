// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/version.h"

#include "onnxruntime_c_api.h"  // NOLINT

namespace sherpa_onnx {

const char *GetGitDate() {
  static const char *date = "Fri Oct 23 14:36:20 2026";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "12b4f4af";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "2.0.0";
  return version;
}

const char *GetOnnxruntimeVersionStr() {
  return OrtGetApiBase()->GetVersionString();
}

}  // namespace sherpa_onnx
