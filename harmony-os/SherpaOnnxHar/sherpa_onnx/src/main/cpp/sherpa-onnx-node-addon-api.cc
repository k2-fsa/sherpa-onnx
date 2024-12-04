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

void InitKeywordSpotting(Napi::Env env, Napi::Object exports);

void InitNonStreamingSpeakerDiarization(Napi::Env env, Napi::Object exports);

#if __OHOS__
void InitUtils(Napi::Env env, Napi::Object exports);
#endif

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
  InitKeywordSpotting(env, exports);
  InitNonStreamingSpeakerDiarization(env, exports);

#if __OHOS__
  InitUtils(env, exports);
#endif

  return exports;
}

#if __OHOS__
NODE_API_MODULE(sherpa_onnx, Init)
#else
NODE_API_MODULE(addon, Init)
#endif
