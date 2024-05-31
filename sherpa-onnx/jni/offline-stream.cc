// sherpa-onnx/jni/offline-stream.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-stream.h"

#include "sherpa-onnx/jni/common.h"

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OfflineStream_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::OfflineStream *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OfflineStream_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto stream = reinterpret_cast<sherpa_onnx::OfflineStream *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  stream->AcceptWaveform(sample_rate, p, n);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}
