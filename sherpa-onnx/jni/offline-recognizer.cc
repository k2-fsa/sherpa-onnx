// sherpa-onnx/jni/offline-recognizer.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-recognizer.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static OfflineRecognizerConfig GetOfflineConfig(JNIEnv *env, jobject config) {
  OfflineRecognizerConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  //---------- decoding ----------
  fid = env->GetFieldID(cls, "decodingMethod", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.decoding_method = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "maxActivePaths", "I");
  ans.max_active_paths = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "hotwordsFile", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.hotwords_file = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "hotwordsScore", "F");
  ans.hotwords_score = env->GetFloatField(config, fid);

  //---------- feat config ----------
  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpa/onnx/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  fid = env->GetFieldID(feat_config_cls, "sampleRate", "I");
  ans.feat_config.sampling_rate = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "featureDim", "I");
  ans.feat_config.feature_dim = env->GetIntField(feat_config, fid);

  //---------- model config ----------
  fid = env->GetFieldID(cls, "modelConfig",
                        "Lcom/k2fsa/sherpa/onnx/OfflineModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  fid = env->GetFieldID(model_config_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");
  ans.model_config.num_threads = env->GetIntField(model_config, fid);

  fid = env->GetFieldID(model_config_cls, "debug", "Z");
  ans.model_config.debug = env->GetBooleanField(model_config, fid);

  fid = env->GetFieldID(model_config_cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.provider = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "modelType", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.model_type = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "modelingUnit", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.modeling_unit = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "bpeVocab", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.bpe_vocab = p;
  env->ReleaseStringUTFChars(s, p);

  // transducer
  fid = env->GetFieldID(model_config_cls, "transducer",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTransducerModelConfig;");
  jobject transducer_config = env->GetObjectField(model_config, fid);
  jclass transducer_config_cls = env->GetObjectClass(transducer_config);

  fid = env->GetFieldID(transducer_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.encoder_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(transducer_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.decoder_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(transducer_config_cls, "joiner", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.joiner_filename = p;
  env->ReleaseStringUTFChars(s, p);

  // paraformer
  fid = env->GetFieldID(model_config_cls, "paraformer",
                        "Lcom/k2fsa/sherpa/onnx/OfflineParaformerModelConfig;");
  jobject paraformer_config = env->GetObjectField(model_config, fid);
  jclass paraformer_config_cls = env->GetObjectClass(paraformer_config);

  fid = env->GetFieldID(paraformer_config_cls, "model", "Ljava/lang/String;");

  s = (jstring)env->GetObjectField(paraformer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.paraformer.model = p;
  env->ReleaseStringUTFChars(s, p);

  // whisper
  fid = env->GetFieldID(model_config_cls, "whisper",
                        "Lcom/k2fsa/sherpa/onnx/OfflineWhisperModelConfig;");
  jobject whisper_config = env->GetObjectField(model_config, fid);
  jclass whisper_config_cls = env->GetObjectClass(whisper_config);

  fid = env->GetFieldID(whisper_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.whisper.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.whisper.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_config_cls, "language", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.whisper.language = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_config_cls, "task", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.whisper.task = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_config_cls, "tailPaddings", "I");
  ans.model_config.whisper.tail_paddings =
      env->GetIntField(whisper_config, fid);

  fid = env->GetFieldID(
      model_config_cls, "nemo",
      "Lcom/k2fsa/sherpa/onnx/OfflineNemoEncDecCtcModelConfig;");
  jobject nemo_config = env->GetObjectField(model_config, fid);
  jclass nemo_config_cls = env->GetObjectClass(nemo_config);

  fid = env->GetFieldID(nemo_config_cls, "model", "Ljava/lang/String;");

  s = (jstring)env->GetObjectField(nemo_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.nemo_ctc.model = p;
  env->ReleaseStringUTFChars(s, p);

  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_newFromAsset(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jobject asset_manager,
                                                          jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
  }
#endif
  auto config = sherpa_onnx::GetOfflineConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::OfflineRecognizer(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_newFromFile(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jobject _config) {
  auto config = sherpa_onnx::GetOfflineConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto model = new sherpa_onnx::OfflineRecognizer(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::OfflineRecognizer *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_createStream(JNIEnv * /*env*/,
                                                          jobject /*obj*/,
                                                          jlong ptr) {
  auto recognizer = reinterpret_cast<sherpa_onnx::OfflineRecognizer *>(ptr);
  std::unique_ptr<sherpa_onnx::OfflineStream> s = recognizer->CreateStream();

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_onnx_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_onnx::OfflineStream *p = s.release();
  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_decode(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong streamPtr) {
  auto recognizer = reinterpret_cast<sherpa_onnx::OfflineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OfflineStream *>(streamPtr);

  recognizer->DecodeStream(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_getResult(JNIEnv *env,
                                                       jobject /*obj*/,
                                                       jlong streamPtr) {
  auto stream = reinterpret_cast<sherpa_onnx::OfflineStream *>(streamPtr);
  sherpa_onnx::OfflineRecognitionResult result = stream->GetResult();

  // [0]: text, jstring
  // [1]: tokens, array of jstring
  // [2]: timestamps, array of float
  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      3, env->FindClass("java/lang/Object"), nullptr);

  jstring text = env->NewStringUTF(result.text.c_str());
  env->SetObjectArrayElement(obj_arr, 0, text);

  jobjectArray tokens_arr = (jobjectArray)env->NewObjectArray(
      result.tokens.size(), env->FindClass("java/lang/String"), nullptr);

  int32_t i = 0;
  for (const auto &t : result.tokens) {
    jstring jtext = env->NewStringUTF(t.c_str());
    env->SetObjectArrayElement(tokens_arr, i, jtext);
    i += 1;
  }

  env->SetObjectArrayElement(obj_arr, 1, tokens_arr);

  jfloatArray timestamps_arr = env->NewFloatArray(result.timestamps.size());
  env->SetFloatArrayRegion(timestamps_arr, 0, result.timestamps.size(),
                           result.timestamps.data());

  env->SetObjectArrayElement(obj_arr, 2, timestamps_arr);

  return obj_arr;
}
