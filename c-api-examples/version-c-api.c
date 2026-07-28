// c-api-examples/version-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation

#include <stdio.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  printf("sherpa-onnx version : %s\n", SherpaOnnxGetVersionStr());
  printf("sherpa-onnx Git SHA1: %s\n", SherpaOnnxGetGitSha1());
  printf("sherpa-onnx Git date: %s\n", SherpaOnnxGetGitDate());
  printf("onnxruntime version : %s\n", SherpaOnnxGetOnnxruntimeVersionStr());

  return 0;
}
