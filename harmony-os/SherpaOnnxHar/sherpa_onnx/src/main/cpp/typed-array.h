// scripts/node-addon-api/src/typed-array.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_HARMONY_OS_SHERPAONNXHAR_SHERPAONNX_SRC_MAIN_CPP_TYPED_ARRAY_H_
#define SHERPA_ONNX_HARMONY_OS_SHERPAONNXHAR_SHERPAONNX_SRC_MAIN_CPP_TYPED_ARRAY_H_

#include "napi.h"  // NOLINT

// Number of elements of a Float32Array.
static inline int32_t GetFloat32ArrayElementLength(Napi::Float32Array samples) {
#if __OHOS__
  // Note(fangjun): For unknown reasons on HarmonyOS, we need to divide it by
  // sizeof(float) here. ElementLength() returns the number of bytes, not the
  // number of elements.
  return samples.ElementLength() / sizeof(float);
#else
  return samples.ElementLength();
#endif
}

// Check that a Float32Array holds exactly `expected` floats; a buffer that a
// getter detached has a length of 0 and is rejected as well. Call this only
// after all property getters have run and before using Data().
static inline bool CheckFloat32ArrayLength(Napi::Env env,
                                           Napi::Float32Array samples,
                                           int64_t expected,
                                           const char *message) {
  if (GetFloat32ArrayElementLength(samples) != expected) {
    Napi::RangeError::New(env, message).ThrowAsJavaScriptException();
    return false;
  }
  return true;
}

#endif  // SHERPA_ONNX_HARMONY_OS_SHERPAONNXHAR_SHERPAONNX_SRC_MAIN_CPP_TYPED_ARRAY_H_
