// sherpa-onnx/jni/wave-reader.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-onnx/csrc/wave-reader.h"

#include <fstream>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

static jobject ReadWaveImpl(JNIEnv *env, std::istream &is,
                            const char *p_filename) {
  bool is_ok = false;
  int32_t sampling_rate = -1;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(is, &sampling_rate, &is_ok);

  if (!is_ok) {
    SHERPA_ONNX_LOGE("Failed to read '%s'", p_filename);
    jclass exception_class = env->FindClass("java/lang/Exception");
    env->ThrowNew(exception_class, "Failed to read wave file.");
    return nullptr;
  }

  jfloatArray samples_arr = env->NewFloatArray(samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, samples.size(), samples.data());

  // Find WaveData class
  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/WaveData");
  if (cls == nullptr) {
    SHERPA_ONNX_LOGE("Failed to find class com/k2fsa/sherpa/onnx/WaveData");
    return nullptr;
  }

  // Get constructor: WaveData(float[] samples, int sampleRate)
  jmethodID ctor = env->GetMethodID(cls, "<init>", "([FI)V");
  if (ctor == nullptr) {
    SHERPA_ONNX_LOGE("Failed to get WaveData constructor");
    return nullptr;
  }

  // Create WaveData object
  jobject obj = env->NewObject(cls, ctor, samples_arr, sampling_rate);

  // Clean up local refs
  env->DeleteLocalRef(samples_arr);

  return obj;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL
Java_com_k2fsa_sherpa_onnx_WaveReader_00024Companion_readWaveFromFile(
    JNIEnv *env, jclass /*cls*/, jstring filename) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);
  std::ifstream is(p_filename, std::ios::binary);

  auto obj = ReadWaveImpl(env, is, p_filename);

  env->ReleaseStringUTFChars(filename, p_filename);

  return obj;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL
Java_com_k2fsa_sherpa_onnx_WaveReader_readWaveFromFile(JNIEnv *env,
                                                       jclass /*obj*/,
                                                       jstring filename) {
  return Java_com_k2fsa_sherpa_onnx_WaveReader_00024Companion_readWaveFromFile(
      env, nullptr, filename);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL
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

  auto obj = ReadWaveImpl(env, is, p_filename);

  env->ReleaseStringUTFChars(filename, p_filename);

  return obj;
}
