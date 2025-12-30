// sherpa-onnx/jni/spoken-language-identification.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/spoken-language-identification.h"

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static SpokenLanguageIdentificationConfig GetSpokenLanguageIdentificationConfig(
    JNIEnv *env, jobject config, bool *ok) {
  SpokenLanguageIdentificationConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid = env->GetFieldID(
      cls, "whisper",
      "Lcom/k2fsa/sherpa/onnx/SpokenLanguageIdentificationWhisperConfig;");

  jobject whisper = env->GetObjectField(config, fid);
  jclass whisper_cls = env->GetObjectClass(whisper);

  SHERPA_ONNX_JNI_READ_STRING(ans.whisper.encoder, encoder, whisper_cls,
                              whisper);

  SHERPA_ONNX_JNI_READ_STRING(ans.whisper.decoder, decoder, whisper_cls,
                              whisper);

  SHERPA_ONNX_JNI_READ_INT(ans.whisper.tail_paddings, tailPaddings, whisper_cls,
                           whisper);

  SHERPA_ONNX_JNI_READ_INT(ans.num_threads, numThreads, cls, config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.debug, debug, cls, config);

  SHERPA_ONNX_JNI_READ_STRING(ans.provider, provider, cls, config);

  *ok = true;
  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_SpokenLanguageIdentification_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif

  bool ok = false;
  auto config =
      sherpa_onnx::GetSpokenLanguageIdentificationConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("spoken language identification newFromAsset config:\n%s",
                   config.ToString().c_str());

  auto slid = new sherpa_onnx::SpokenLanguageIdentification(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);
  SHERPA_ONNX_LOGE("slid %p", slid);

  return (jlong)slid;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_SpokenLanguageIdentification_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  bool ok = false;
  auto config =
      sherpa_onnx::GetSpokenLanguageIdentificationConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("SpokenLanguageIdentification newFromFile config:\n%s",
                   config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto tagger = new sherpa_onnx::SpokenLanguageIdentification(config);

  return (jlong)tagger;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_SpokenLanguageIdentification_delete(JNIEnv * /*env*/,
                                                               jobject /*obj*/,
                                                               jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::SpokenLanguageIdentification *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_SpokenLanguageIdentification_createStream(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  auto slid =
      reinterpret_cast<sherpa_onnx::SpokenLanguageIdentification *>(ptr);
  std::unique_ptr<sherpa_onnx::OfflineStream> s = slid->CreateStream();

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_onnx_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_onnx::OfflineStream *p = s.release();
  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_onnx_SpokenLanguageIdentification_compute(JNIEnv *env,
                                                                jobject /*obj*/,
                                                                jlong ptr,
                                                                jlong s_ptr) {
  sherpa_onnx::SpokenLanguageIdentification *slid =
      reinterpret_cast<sherpa_onnx::SpokenLanguageIdentification *>(ptr);
  sherpa_onnx::OfflineStream *s =
      reinterpret_cast<sherpa_onnx::OfflineStream *>(s_ptr);
  std::string lang = slid->Compute(s);
  return env->NewStringUTF(lang.c_str());
}
