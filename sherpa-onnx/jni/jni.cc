// sherpa-onnx/jni/jni.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation
//                2022       Pingfeng Luo
//                2023       Zhaoming

#include <fstream>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/wave-writer.h"
#include "sherpa-onnx/jni/common.h"

// see
// https://stackoverflow.com/questions/29043872/android-jni-return-multiple-variables
jobject NewInteger(JNIEnv *env, int32_t value) {
  jclass cls = env->FindClass("java/lang/Integer");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(I)V");
  return env->NewObject(cls, constructor, value);
}

jobject NewFloat(JNIEnv *env, float value) {
  jclass cls = env->FindClass("java/lang/Float");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(F)V");
  return env->NewObject(cls, constructor, value);
}

#if 0
SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_decodeStreams(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jlongArray ss_ptr,
                                                          jint stream_size) {
  sherpa_onnx::OnlineRecognizer *model =
      reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  jlong *p = env->GetLongArrayElements(ss_ptr, nullptr);
  jsize n = env->GetArrayLength(ss_ptr);
  std::vector<sherpa_onnx::OnlineStream *> p_ss(n);
  for (int32_t i = 0; i != n; ++i) {
    p_ss[i] = reinterpret_cast<sherpa_onnx::OnlineStream *>(p[i]);
  }

  model->DecodeStreams(p_ss.data(), n);
  env->ReleaseLongArrayElements(ss_ptr, p, JNI_ABORT);
}
#endif
