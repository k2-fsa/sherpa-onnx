// sherpa-onnx/jni/online-speech-denoiser.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-speech-denoiser.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"
#include "sherpa-onnx/jni/speech-denoiser.h"

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif

  bool ok = false;
  auto config = sherpa_onnx::GetOnlineSpeechDenoiserConfig(env, _config, &ok);
  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto speech_denoiser = new sherpa_onnx::OnlineSpeechDenoiser(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return reinterpret_cast<jlong>(speech_denoiser);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_newFromFile(JNIEnv *env,
                                                            jobject /*obj*/,
                                                            jobject _config) {
  return SafeJNI(
      env, "OnlineSpeechDenoiser_newFromFile",
      [&]() -> jlong {
        bool ok = false;
        auto config =
            sherpa_onnx::GetOnlineSpeechDenoiserConfig(env, _config, &ok);

        if (!ok) {
          SHERPA_ONNX_LOGE("Please read the error message carefully");
          return 0;
        }

        SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

        if (!config.Validate()) {
          SHERPA_ONNX_LOGE("Errors found in config!");
          return 0;
        }

        auto speech_denoiser = new sherpa_onnx::OnlineSpeechDenoiser(config);
        return reinterpret_cast<jlong>(speech_denoiser);
      },
      static_cast<jlong>(0));
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::OnlineSpeechDenoiser *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_getSampleRate(JNIEnv * /*env*/,
                                                              jobject /*obj*/,
                                                              jlong ptr) {
  return reinterpret_cast<sherpa_onnx::OnlineSpeechDenoiser *>(ptr)
      ->GetSampleRate();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_getFrameShiftInSamples(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_onnx::OnlineSpeechDenoiser *>(ptr)
      ->GetFrameShiftInSamples();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_run(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto speech_denoiser =
      reinterpret_cast<sherpa_onnx::OnlineSpeechDenoiser *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  auto denoised = speech_denoiser->Run(p, n, sample_rate);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  return sherpa_onnx::NewDenoisedAudio(env, denoised);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_flush(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto speech_denoiser =
      reinterpret_cast<sherpa_onnx::OnlineSpeechDenoiser *>(ptr);
  auto denoised = speech_denoiser->Flush();
  return sherpa_onnx::NewDenoisedAudio(env, denoised);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_reset(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  reinterpret_cast<sherpa_onnx::OnlineSpeechDenoiser *>(ptr)->Reset();
}
