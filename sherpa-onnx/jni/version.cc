// sherpa-onnx/jni/version.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-onnx/csrc/version.h"

#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_onnx_VersionInfo_00024Companion_getVersionStr2(
    JNIEnv *env, jclass /*cls*/) {
  return env->NewStringUTF(GetVersionStr());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_onnx_VersionInfo_00024Companion_getGitSha12(
    JNIEnv *env, jclass /*cls*/) {
  return env->NewStringUTF(GetGitSha1());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_onnx_VersionInfo_00024Companion_getGitDate2(
    JNIEnv *env, jclass /*cls*/) {
  return env->NewStringUTF(GetGitDate());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpa_onnx_VersionInfo_getVersionStr2(
    JNIEnv *env, jclass /*cls*/) {
  return env->NewStringUTF(GetVersionStr());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpa_onnx_VersionInfo_getGitSha12(
    JNIEnv *env, jclass /*cls*/) {
  return env->NewStringUTF(GetGitSha1());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpa_onnx_VersionInfo_getGitDate2(
    JNIEnv *env, jclass /*cls*/) {
  return env->NewStringUTF(GetGitDate());
}

}  // namespace sherpa_onnx
