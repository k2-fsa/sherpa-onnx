// scripts/node-addon-api/src/sherpa-onnx-node-addon-api.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "napi.h"  // NOLINT

void InitStreamingAsr(Napi::Env env, Napi::Object exports);

void InitNonStreamingAsr(Napi::Env env, Napi::Object exports);

void InitNonStreamingTts(Napi::Env env, Napi::Object exports);

void InitVad(Napi::Env env, Napi::Object exports);

void InitWaveReader(Napi::Env env, Napi::Object exports);

void InitWaveWriter(Napi::Env env, Napi::Object exports);

void InitSpokenLanguageID(Napi::Env env, Napi::Object exports);

void InitSpeakerID(Napi::Env env, Napi::Object exports);

void InitAudioTagging(Napi::Env env, Napi::Object exports);

void InitPunctuation(Napi::Env env, Napi::Object exports);

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  InitStreamingAsr(env, exports);
  InitNonStreamingAsr(env, exports);
  InitNonStreamingTts(env, exports);
  InitVad(env, exports);
  InitWaveReader(env, exports);
  InitWaveWriter(env, exports);
  InitSpokenLanguageID(env, exports);
  InitSpeakerID(env, exports);
  InitAudioTagging(env, exports);
  InitPunctuation(env, exports);

  return exports;
}

NODE_API_MODULE(addon, Init)
