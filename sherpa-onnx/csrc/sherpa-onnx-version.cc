// sherpa-onnx/csrc/sherpa-onnx-version.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <stdio.h>

#include <cstdint>

#include "sherpa-onnx/csrc/version.h"

int32_t main() {
  printf("sherpa-onnx version : %s\n", sherpa_onnx::GetVersionStr());
  printf("sherpa-onnx Git SHA1: %s\n", sherpa_onnx::GetGitSha1());
  printf("sherpa-onnx Git date: %s\n", sherpa_onnx::GetGitDate());

  return 0;
}
