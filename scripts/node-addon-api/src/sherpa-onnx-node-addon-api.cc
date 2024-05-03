// scripts/node-addon-api/src/sherpa-onnx-node-addon-api.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "napi.h"  // NOLINT

Napi::Object InitStreamingAsr(Napi::Env env, Napi::Object exports);
void InitWaveReader(Napi::Env env, Napi::Object exports);

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  InitStreamingAsr(env, exports);
  InitWaveReader(env, exports);

  return exports;
}

NODE_API_MODULE(addon, Init)
