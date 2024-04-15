// sherpa-onnx/jni/common.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_JNI_COMMON_H_
#define SHERPA_ONNX_JNI_COMMON_H_

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

// If you use ndk, you can find "jni.h" inside
// android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
#include "jni.h"  // NOLINT

#define SHERPA_ONNX_EXTERN_C extern "C"

// defined in jni.cc
jobject NewInteger(JNIEnv *env, int32_t value);
jobject NewFloat(JNIEnv *env, float value);

#endif  // SHERPA_ONNX_JNI_COMMON_H_
