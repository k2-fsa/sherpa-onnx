// sherpa-onnx/jni/offline-speech-denoiser.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-onnx/csrc/offline-speech-denoiser.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/wave-writer.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {
static OfflineSpeechDenoiserConfig GetOfflineSpeechDenoiserConfig(
    JNIEnv *env, jobject config) {
  OfflineSpeechDenoiserConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  fid = env->GetFieldID(
      cls, "model", "Lcom/k2fsa/sherpa/onnx/OfflineSpeechDenoiserModelConfig;");
  jobject model = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model);

  fid = env->GetFieldID(
      model_config_cls, "gtcrn",
      "Lcom/k2fsa/sherpa/onnx/OfflineSpeechDenoiserGtcrnModelConfig;");
  jobject gtcrn = env->GetObjectField(model, fid);
  jclass gtcrn_cls = env->GetObjectClass(gtcrn);

  fid = env->GetFieldID(gtcrn_cls, "model", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(gtcrn, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.model.gtcrn.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");
  ans.model.num_threads = env->GetIntField(model, fid);

  fid = env->GetFieldID(model_config_cls, "debug", "Z");
  ans.model.debug = env->GetBooleanField(model, fid);

  fid = env->GetFieldID(model_config_cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.provider = p;
  env->ReleaseStringUTFChars(s, p);

  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeechDenoiser_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  auto config = sherpa_onnx::GetOfflineSpeechDenoiserConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto speech_denoiser = new sherpa_onnx::OfflineSpeechDenoiser(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)speech_denoiser;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeechDenoiser_newFromFile(JNIEnv *env,
                                                             jobject /*obj*/,
                                                             jobject _config) {
  return SafeJNI(
      env, "OfflineSpeechDenoiser_newFromFile",
      [&]() -> jlong {
        auto config = sherpa_onnx::GetOfflineSpeechDenoiserConfig(env, _config);
        SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

        if (!config.Validate()) {
          SHERPA_ONNX_LOGE("Errors found in config!");
        }

        auto speech_denoiser = new sherpa_onnx::OfflineSpeechDenoiser(config);
        return reinterpret_cast<jlong>(speech_denoiser);
      },
      (jlong)0);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OfflineSpeechDenoiser_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::OfflineSpeechDenoiser *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeechDenoiser_getSampleRate(JNIEnv * /*env*/,
                                                               jobject /*obj*/,
                                                               jlong ptr) {
  return reinterpret_cast<sherpa_onnx::OfflineSpeechDenoiser *>(ptr)
      ->GetSampleRate();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL Java_com_k2fsa_sherpa_onnx_OfflineSpeechDenoiser_run(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto speech_denoiser =
      reinterpret_cast<sherpa_onnx::OfflineSpeechDenoiser *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  auto denoised = speech_denoiser->Run(p, n, sample_rate);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/DenoisedAudio");
  if (cls == nullptr) {
    SHERPA_ONNX_LOGE("Failed to get class for DenoisedAudio");
    return nullptr;
  }

  // https://javap.yawk.at/
  jmethodID constructor = env->GetMethodID(cls, "<init>", "([FI)V");
  if (constructor == nullptr) {
    SHERPA_ONNX_LOGE("Failed to get constructor for DenoisedAudio");
    return nullptr;
  }

  jfloatArray samples_arr = env->NewFloatArray(denoised.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, denoised.samples.size(),
                           denoised.samples.data());

  return env->NewObject(cls, constructor, samples_arr, denoised.sample_rate);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL Java_com_k2fsa_sherpa_onnx_DenoisedAudio_saveImpl(
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
