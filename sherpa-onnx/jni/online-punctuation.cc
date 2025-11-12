// sherpa-onnx/jni/online-punctuation.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-punctuation.h"

#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static OnlinePunctuationConfig GetOnlinePunctuationConfig(JNIEnv *env,
                                                          jobject config,
                                                          bool *ok) {
  OnlinePunctuationConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  fid = env->GetFieldID(cls, "model",
                        "Lcom/k2fsa/sherpa/onnx/OnlinePunctuationModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.cnn_bilstm, cnnBilstm, model_config_cls,
                              model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.bpe_vocab, bpeVocab, model_config_cls,
                              model_config);

  SHERPA_ONNX_JNI_READ_INT(ans.model.num_threads, numThreads, model_config_cls,
                           model_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.model.debug, debug, model_config_cls,
                            model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.provider, provider, model_config_cls,
                              model_config);

  *ok = true;
  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OnlinePunctuation_newFromAsset(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jobject asset_manager,
                                                          jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  bool ok = false;
  auto config = sherpa_onnx::GetOnlinePunctuationConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto model = new sherpa_onnx::OnlinePunctuation(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OnlinePunctuation_newFromFile(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jobject _config) {
  bool ok = false;
  auto config = sherpa_onnx::GetOnlinePunctuationConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto model = new sherpa_onnx::OnlinePunctuation(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlinePunctuation_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::OnlinePunctuation *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_onnx_OnlinePunctuation_addPunctuation(JNIEnv *env,
                                                            jobject /*obj*/,
                                                            jlong ptr,
                                                            jstring text) {
  auto punct = reinterpret_cast<const sherpa_onnx::OnlinePunctuation *>(ptr);

  const char *ptext = env->GetStringUTFChars(text, nullptr);

  std::string result = punct->AddPunctuationWithCase(ptext);

  env->ReleaseStringUTFChars(text, ptext);

  return env->NewStringUTF(result.c_str());
}
