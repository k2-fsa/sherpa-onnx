// sherpa-onnx/jni/speech-denoiser.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/jni/speech-denoiser.h"

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

OfflineSpeechDenoiserModelConfig GetOfflineSpeechDenoiserModelConfig(
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

OfflineSpeechDenoiserConfig GetOfflineSpeechDenoiserConfig(JNIEnv *env,
                                                           jobject config,
                                                           bool *ok) {
  OfflineSpeechDenoiserConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid = env->GetFieldID(
      cls, "model", "Lcom/k2fsa/sherpa/onnx/OfflineSpeechDenoiserModelConfig;");
  jobject model = env->GetObjectField(config, fid);

  ans.model = GetOfflineSpeechDenoiserModelConfig(env, model, ok);
  return ans;
}

OnlineSpeechDenoiserConfig GetOnlineSpeechDenoiserConfig(JNIEnv *env,
                                                         jobject config,
                                                         bool *ok) {
  OnlineSpeechDenoiserConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid = env->GetFieldID(
      cls, "model", "Lcom/k2fsa/sherpa/onnx/OfflineSpeechDenoiserModelConfig;");
  jobject model = env->GetObjectField(config, fid);

  ans.model = GetOfflineSpeechDenoiserModelConfig(env, model, ok);
  return ans;
}

jobject NewDenoisedAudio(JNIEnv *env, const DenoisedAudio &denoised) {
  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/DenoisedAudio");
  if (cls == nullptr) {
    SHERPA_ONNX_LOGE("Failed to get class for DenoisedAudio");
    return nullptr;
  }

  jmethodID constructor = env->GetMethodID(cls, "<init>", "([FI)V");
  if (constructor == nullptr) {
    SHERPA_ONNX_LOGE("Failed to get constructor for DenoisedAudio");
    env->DeleteLocalRef(cls);
    return nullptr;
  }

  jfloatArray samples_arr = env->NewFloatArray(denoised.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, denoised.samples.size(),
                           denoised.samples.data());

  jobject obj =
      env->NewObject(cls, constructor, samples_arr, denoised.sample_rate);
  env->DeleteLocalRef(cls);
  env->DeleteLocalRef(samples_arr);
  return obj;
}

}  // namespace sherpa_onnx
