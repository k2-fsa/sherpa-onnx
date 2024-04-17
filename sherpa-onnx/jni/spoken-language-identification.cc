// sherpa-onnx/jni/spoken-language-identification.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/spoken-language-identification.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static SpokenLanguageIdentificationConfig GetSpokenLanguageIdentificationConfig(
    JNIEnv *env, jobject config) {
  SpokenLanguageIdentificationConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid = env->GetFieldID(
      cls, "whisper",
      "Lcom/k2fsa/sherpa/onnx/SpokenLanguageIdentificationWhisperConfig;");

  jobject whisper = env->GetObjectField(config, fid);
  jclass whisper_cls = env->GetObjectClass(whisper);

  fid = env->GetFieldID(whisper_cls, "encoder", "Ljava/lang/String;");

  jstring s = (jstring)env->GetObjectField(whisper, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.whisper.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.whisper.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_cls, "tailPaddings", "I");
  ans.whisper.tail_paddings = env->GetIntField(whisper, fid);

  fid = env->GetFieldID(cls, "numThreads", "I");
  ans.numThreads = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "debug", "Z");
  ans.debug = env->GetBooleanField(config, fid);

  fid = env->GetFieldID(cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.provider = p;
  env->ReleaseStringUTFChars(s, p);

  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_SpokenLanguageIdentification_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config =
      sherpa_onnx::GetSpokenLanguageIdentificationConfig(env, _config);
  SHERPA_ONNX_LOGE("SpokenLanguageIdentification newFromFile config:\n%s",
                   config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto tagger = new sherpa_onnx::SpokenLanguageIdentification(config);

  return (jlong)tagger;
}
