// sherpa-onnx/jni/online-speech-denoiser.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-speech-denoiser.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static OfflineSpeechDenoiserModelConfig GetOfflineSpeechDenoiserModelConfig(
    JNIEnv *env, jobject model, bool *ok) {
  OfflineSpeechDenoiserModelConfig ans;

  jclass model_config_cls = env->GetObjectClass(model);
  jfieldID fid;

  fid = env->GetFieldID(
      model_config_cls, "gtcrn",
      "Lcom/k2fsa/sherpa/onnx/OfflineSpeechDenoiserGtcrnModelConfig;");
  jobject gtcrn = env->GetObjectField(model, fid);
  jclass gtcrn_cls = env->GetObjectClass(gtcrn);

  SHERPA_ONNX_JNI_READ_STRING(ans.gtcrn.model, model, gtcrn_cls, gtcrn);

  fid = env->GetFieldID(
      model_config_cls, "dpdfnet",
      "Lcom/k2fsa/sherpa/onnx/OfflineSpeechDenoiserDpdfNetModelConfig;");
  jobject dpdfnet = env->GetObjectField(model, fid);
  jclass dpdfnet_cls = env->GetObjectClass(dpdfnet);

  SHERPA_ONNX_JNI_READ_STRING(ans.dpdfnet.model, model, dpdfnet_cls, dpdfnet);

  SHERPA_ONNX_JNI_READ_INT(ans.num_threads, numThreads, model_config_cls,
                           model);

  SHERPA_ONNX_JNI_READ_BOOL(ans.debug, debug, model_config_cls, model);

  SHERPA_ONNX_JNI_READ_STRING(ans.provider, provider, model_config_cls, model);

  *ok = true;
  return ans;
}

static OnlineSpeechDenoiserConfig GetOnlineSpeechDenoiserConfig(
    JNIEnv *env, jobject config, bool *ok) {
  OnlineSpeechDenoiserConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid = env->GetFieldID(
      cls, "model", "Lcom/k2fsa/sherpa/onnx/OfflineSpeechDenoiserModelConfig;");
  jobject model = env->GetObjectField(config, fid);

  ans.model = GetOfflineSpeechDenoiserModelConfig(env, model, ok);
  return ans;
}

static jobject NewDenoisedAudio(JNIEnv *env, const DenoisedAudio &denoised) {
  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/DenoisedAudio");
  if (cls == nullptr) {
    SHERPA_ONNX_LOGE("Failed to get class for DenoisedAudio");
    return nullptr;
  }

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

}  // namespace sherpa_onnx

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
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_getFrameShiftInSamples(
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
JNIEXPORT jobject JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineSpeechDenoiser_flush(JNIEnv *env,
                                                      jobject /*obj*/,
                                                      jlong ptr) {
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
