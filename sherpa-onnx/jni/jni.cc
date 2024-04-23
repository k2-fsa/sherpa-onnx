// sherpa-onnx/jni/jni.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation
//                2022       Pingfeng Luo
//                2023       Zhaoming

#include <fstream>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/wave-reader.h"
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

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL Java_com_k2fsa_sherpa_onnx_GeneratedAudio_saveImpl(
    JNIEnv *env, jobject /*obj*/, jstring filename, jfloatArray samples,
    jint sample_rate) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  bool ok = sherpa_onnx::WriteWave(p_filename, sample_rate, p, n);

  env->ReleaseStringUTFChars(filename, p_filename);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  return ok;
}

static jobjectArray ReadWaveImpl(JNIEnv *env, std::istream &is,
                                 const char *p_filename) {
  bool is_ok = false;
  int32_t sampling_rate = -1;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(is, &sampling_rate, &is_ok);

  if (!is_ok) {
    SHERPA_ONNX_LOGE("Failed to read %s", p_filename);
    exit(-1);
  }

  jfloatArray samples_arr = env->NewFloatArray(samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, samples.size(), samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, sampling_rate));

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_WaveReader_00024Companion_readWaveFromFile(
    JNIEnv *env, jclass /*cls*/, jstring filename) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);
  std::ifstream is(p_filename, std::ios::binary);

  auto obj_arr = ReadWaveImpl(env, is, p_filename);

  env->ReleaseStringUTFChars(filename, p_filename);

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_WaveReader_00024Companion_readWaveFromAsset(
    JNIEnv *env, jclass /*cls*/, jobject asset_manager, jstring filename) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    exit(-1);
  }
  std::vector<char> buffer = sherpa_onnx::ReadFile(mgr, p_filename);

  std::istrstream is(buffer.data(), buffer.size());
#else
  std::ifstream is(p_filename, std::ios::binary);
#endif

  auto obj_arr = ReadWaveImpl(env, is, p_filename);

  env->ReleaseStringUTFChars(filename, p_filename);

  return obj_arr;
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
