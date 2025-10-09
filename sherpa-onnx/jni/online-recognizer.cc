// sherpa-onnx/jni/online-recognizer.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-recognizer.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

OnlineModelConfig GetOnlineModelConfig(JNIEnv *env, jclass model_config_cls,
                                       jobject model_config, bool *ok) {
  OnlineModelConfig ans;

  auto fid =
      env->GetFieldID(model_config_cls, "transducer",
                      "Lcom/k2fsa/sherpa/onnx/OnlineTransducerModelConfig;");
  jobject transducer_config = env->GetObjectField(model_config, fid);
  jclass transducer_config_cls = env->GetObjectClass(transducer_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.transducer.encoder, encoder,
                              transducer_config_cls, transducer_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.transducer.decoder, decoder,
                              transducer_config_cls, transducer_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.transducer.joiner, joiner,
                              transducer_config_cls, transducer_config);

  fid = env->GetFieldID(model_config_cls, "paraformer",
                        "Lcom/k2fsa/sherpa/onnx/OnlineParaformerModelConfig;");
  jobject paraformer_config = env->GetObjectField(model_config, fid);
  jclass paraformer_config_cls = env->GetObjectClass(paraformer_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.paraformer.encoder, encoder,
                              paraformer_config_cls, paraformer_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.paraformer.decoder, decoder,
                              paraformer_config_cls, paraformer_config);

  fid =
      env->GetFieldID(model_config_cls, "zipformer2Ctc",
                      "Lcom/k2fsa/sherpa/onnx/OnlineZipformer2CtcModelConfig;");
  jobject zipformer2_ctc_config = env->GetObjectField(model_config, fid);
  jclass zipformer2_ctc_config_cls = env->GetObjectClass(zipformer2_ctc_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.zipformer2_ctc.model, model,
                              zipformer2_ctc_config_cls, zipformer2_ctc_config);

  fid = env->GetFieldID(model_config_cls, "neMoCtc",
                        "Lcom/k2fsa/sherpa/onnx/OnlineNeMoCtcModelConfig;");
  jobject nemo_ctc_config = env->GetObjectField(model_config, fid);
  jclass nemo_ctc_config_cls = env->GetObjectClass(nemo_ctc_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.nemo_ctc.model, model, nemo_ctc_config_cls,
                              nemo_ctc_config);

  fid = env->GetFieldID(model_config_cls, "toneCtc",
                        "Lcom/k2fsa/sherpa/onnx/OnlineToneCtcModelConfig;");
  jobject t_one_ctc_config = env->GetObjectField(model_config, fid);
  jclass t_one_ctc_config_cls = env->GetObjectClass(t_one_ctc_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.t_one_ctc.model, model, t_one_ctc_config_cls,
                              t_one_ctc_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.tokens, tokens, model_config_cls,
                              model_config);

  SHERPA_ONNX_JNI_READ_INT(ans.num_threads, numThreads, model_config_cls,
                           model_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.debug, debug, model_config_cls, model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.provider_config.provider, provider,
                              model_config_cls, model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_type, modelType, model_config_cls,
                              model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.modeling_unit, modelingUnit, model_config_cls,
                              model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.bpe_vocab, bpeVocab, model_config_cls,
                              model_config);

  *ok = true;
  return ans;
}

static OnlineRecognizerConfig GetConfig(JNIEnv *env, jobject config, bool *ok) {
  OnlineRecognizerConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  // https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html
  // https://courses.cs.washington.edu/courses/cse341/99wi/java/tutorial/native1.1/implementing/field.html

  SHERPA_ONNX_JNI_READ_STRING(ans.decoding_method, decodingMethod, cls, config);

  SHERPA_ONNX_JNI_READ_INT(ans.max_active_paths, maxActivePaths, cls, config);

  SHERPA_ONNX_JNI_READ_STRING(ans.hotwords_file, hotwordsFile, cls, config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.hotwords_score, hotwordsScore, cls, config);

  SHERPA_ONNX_JNI_READ_STRING(ans.rule_fsts, ruleFsts, cls, config);

  SHERPA_ONNX_JNI_READ_STRING(ans.rule_fars, ruleFars, cls, config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.blank_penalty, blankPenalty, cls, config);

  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpa/onnx/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  SHERPA_ONNX_JNI_READ_INT(ans.feat_config.sampling_rate, sampleRate,
                           feat_config_cls, feat_config);

  SHERPA_ONNX_JNI_READ_INT(ans.feat_config.feature_dim, featureDim,
                           feat_config_cls, feat_config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.feat_config.dither, dither, feat_config_cls,
                             feat_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.enable_endpoint, enableEndpoint, cls, config);

  fid = env->GetFieldID(cls, "endpointConfig",
                        "Lcom/k2fsa/sherpa/onnx/EndpointConfig;");
  jobject endpoint_config = env->GetObjectField(config, fid);
  jclass endpoint_config_cls = env->GetObjectClass(endpoint_config);

  fid = env->GetFieldID(endpoint_config_cls, "rule1",
                        "Lcom/k2fsa/sherpa/onnx/EndpointRule;");
  jobject rule1 = env->GetObjectField(endpoint_config, fid);
  jclass rule_class = env->GetObjectClass(rule1);

  fid = env->GetFieldID(endpoint_config_cls, "rule2",
                        "Lcom/k2fsa/sherpa/onnx/EndpointRule;");
  jobject rule2 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(endpoint_config_cls, "rule3",
                        "Lcom/k2fsa/sherpa/onnx/EndpointRule;");
  jobject rule3 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(rule_class, "mustContainNonSilence", "Z");
  ans.endpoint_config.rule1.must_contain_nonsilence =
      env->GetBooleanField(rule1, fid);
  ans.endpoint_config.rule2.must_contain_nonsilence =
      env->GetBooleanField(rule2, fid);
  ans.endpoint_config.rule3.must_contain_nonsilence =
      env->GetBooleanField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minTrailingSilence", "F");
  ans.endpoint_config.rule1.min_trailing_silence =
      env->GetFloatField(rule1, fid);
  ans.endpoint_config.rule2.min_trailing_silence =
      env->GetFloatField(rule2, fid);
  ans.endpoint_config.rule3.min_trailing_silence =
      env->GetFloatField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minUtteranceLength", "F");
  ans.endpoint_config.rule1.min_utterance_length =
      env->GetFloatField(rule1, fid);
  ans.endpoint_config.rule2.min_utterance_length =
      env->GetFloatField(rule2, fid);
  ans.endpoint_config.rule3.min_utterance_length =
      env->GetFloatField(rule3, fid);

  //---------- model config ----------
  fid = env->GetFieldID(cls, "modelConfig",
                        "Lcom/k2fsa/sherpa/onnx/OnlineModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  ans.model_config =
      GetOnlineModelConfig(env, model_config_cls, model_config, ok);

  if (!*ok) {
    return ans;
  }

  *ok = false;

  //---------- rnn lm model config ----------
  fid = env->GetFieldID(cls, "lmConfig",
                        "Lcom/k2fsa/sherpa/onnx/OnlineLMConfig;");
  jobject lm_model_config = env->GetObjectField(config, fid);
  jclass lm_model_config_cls = env->GetObjectClass(lm_model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.lm_config.model, model, lm_model_config_cls,
                              lm_model_config);
  SHERPA_ONNX_JNI_READ_FLOAT(ans.lm_config.scale, scale, lm_model_config_cls,
                             lm_model_config);

  fid = env->GetFieldID(cls, "ctcFstDecoderConfig",
                        "Lcom/k2fsa/sherpa/onnx/OnlineCtcFstDecoderConfig;");

  jobject fst_decoder_config = env->GetObjectField(config, fid);
  jclass fst_decoder_config_cls = env->GetObjectClass(fst_decoder_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.ctc_fst_decoder_config.graph, graph,
                              fst_decoder_config_cls, fst_decoder_config);

  SHERPA_ONNX_JNI_READ_INT(ans.ctc_fst_decoder_config.max_active, maxActive,
                           fst_decoder_config_cls, fst_decoder_config);

  fid = env->GetFieldID(cls, "hr",
                        "Lcom/k2fsa/sherpa/onnx/HomophoneReplacerConfig;");
  jobject hr_config = env->GetObjectField(config, fid);
  jclass hr_config_cls = env->GetObjectClass(hr_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.hr.dict_dir, dictDir, hr_config_cls,
                              hr_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.hr.lexicon, lexicon, hr_config_cls,
                              hr_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.hr.rule_fsts, ruleFsts, hr_config_cls,
                              hr_config);

  *ok = true;
  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_newFromAsset(JNIEnv *env,
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
  auto config = sherpa_onnx::GetConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  auto str_vec = sherpa_onnx::SplitString(config.ToString(), 128);
  for (const auto &s : str_vec) {
    SHERPA_ONNX_LOGE("%s", s.c_str());
  }

  auto recognizer = new sherpa_onnx::OnlineRecognizer(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)recognizer;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  bool ok = false;
  auto config = sherpa_onnx::GetConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  auto str_vec = sherpa_onnx::SplitString(config.ToString(), 128);
  for (const auto &s : str_vec) {
    SHERPA_ONNX_LOGE("%s", s.c_str());
  }

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto recognizer = new sherpa_onnx::OnlineRecognizer(config);

  return (jlong)recognizer;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_reset(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(stream_ptr);
  recognizer->Reset(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_isReady(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(stream_ptr);

  return recognizer->IsReady(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_isEndpoint(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(stream_ptr);

  return recognizer->IsEndpoint(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_decode(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(stream_ptr);

  recognizer->DecodeStream(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_decodeStreams(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlongArray stream_ptrs) {
  auto recognizer = reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);

  jlong *p = env->GetLongArrayElements(stream_ptrs, nullptr);
  jsize n = env->GetArrayLength(stream_ptrs);

  auto ss = reinterpret_cast<sherpa_onnx::OnlineStream **>(p);

  recognizer->DecodeStreams(ss, n);

  env->ReleaseLongArrayElements(stream_ptrs, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_createStream(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jlong ptr,
                                                         jstring hotwords) {
  auto recognizer = reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);

  const char *p = env->GetStringUTFChars(hotwords, nullptr);
  std::unique_ptr<sherpa_onnx::OnlineStream> stream;

  if (strlen(p) == 0) {
    stream = recognizer->CreateStream();
  } else {
    stream = recognizer->CreateStream(p);
  }

  env->ReleaseStringUTFChars(hotwords, p);

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_onnx_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_onnx::OnlineStream *ans = stream.release();
  return (jlong)ans;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_getResult(JNIEnv *env,
                                                      jobject /*obj*/,
                                                      jlong ptr,
                                                      jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(stream_ptr);

  sherpa_onnx::OnlineRecognizerResult result = recognizer->GetResult(stream);

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
