// scripts/node-addon-api/src/version.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include <sstream>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

void InitVersion(Napi::Env env, Napi::Object exports) {
  Napi::String version = Napi::String::New(env, SherpaOnnxGetVersionStr());
  Napi::String git_sha1 = Napi::String::New(env, SherpaOnnxGetGitSha1());
  Napi::String git_date = Napi::String::New(env, SherpaOnnxGetGitDate());

  exports.Set(Napi::String::New(env, "version"), version);
  exports.Set(Napi::String::New(env, "gitSha1"), git_sha1);
  exports.Set(Napi::String::New(env, "gitDate"), git_date);
}
