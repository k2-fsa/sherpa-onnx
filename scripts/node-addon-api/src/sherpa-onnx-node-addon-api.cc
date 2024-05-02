// scripts/node-addon-api/src/sherpa-onnx-node-addon-api.cc
#include "napi.h"

Napi::Object InitStreamingAsr(Napi::Env env, Napi::Object exports);

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  InitStreamingAsr(env, exports);

  return exports;
}

NODE_API_MODULE(addon, Init)
