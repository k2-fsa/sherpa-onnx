// sherpa-onnx/jni/offline-recognizer.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-recognizer.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"
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

  fid = env->GetFieldID(cls, "ruleFsts", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.rule_fsts = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "ruleFars", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.rule_fars = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "blankPenalty", "F");
  ans.blank_penalty = env->GetFloatField(config, fid);

  //---------- feat config ----------
  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpa/onnx/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  fid = env->GetFieldID(feat_config_cls, "sampleRate", "I");
  ans.feat_config.sampling_rate = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "featureDim", "I");
  ans.feat_config.feature_dim = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "dither", "F");
  ans.feat_config.dither = env->GetFloatField(feat_config, fid);

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

  // FireRedAsr
  fid = env->GetFieldID(model_config_cls, "fireRedAsr",
                        "Lcom/k2fsa/sherpa/onnx/OfflineFireRedAsrModelConfig;");
  jobject fire_red_asr_config = env->GetObjectField(model_config, fid);
  jclass fire_red_asr_config_cls = env->GetObjectClass(fire_red_asr_config);

  fid =
      env->GetFieldID(fire_red_asr_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(fire_red_asr_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.fire_red_asr.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid =
      env->GetFieldID(fire_red_asr_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(fire_red_asr_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.fire_red_asr.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  // moonshine
  fid = env->GetFieldID(model_config_cls, "moonshine",
                        "Lcom/k2fsa/sherpa/onnx/OfflineMoonshineModelConfig;");
  jobject moonshine_config = env->GetObjectField(model_config, fid);
  jclass moonshine_config_cls = env->GetObjectClass(moonshine_config);

  fid = env->GetFieldID(moonshine_config_cls, "preprocessor",
                        "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(moonshine_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.moonshine.preprocessor = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(moonshine_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(moonshine_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.moonshine.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(moonshine_config_cls, "uncachedDecoder",
                        "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(moonshine_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.moonshine.uncached_decoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(moonshine_config_cls, "cachedDecoder",
                        "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(moonshine_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.moonshine.cached_decoder = p;
  env->ReleaseStringUTFChars(s, p);

  // sense voice
  fid = env->GetFieldID(model_config_cls, "senseVoice",
                        "Lcom/k2fsa/sherpa/onnx/OfflineSenseVoiceModelConfig;");
  jobject sense_voice_config = env->GetObjectField(model_config, fid);
  jclass sense_voice_config_cls = env->GetObjectClass(sense_voice_config);

  fid = env->GetFieldID(sense_voice_config_cls, "model", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(sense_voice_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.sense_voice.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid =
      env->GetFieldID(sense_voice_config_cls, "language", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(sense_voice_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.sense_voice.language = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(sense_voice_config_cls, "useInverseTextNormalization",
                        "Z");
  ans.model_config.sense_voice.use_itn =
      env->GetBooleanField(sense_voice_config, fid);

  // nemo
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

  // zipformer ctc
  fid =
      env->GetFieldID(model_config_cls, "zipformerCtc",
                      "Lcom/k2fsa/sherpa/onnx/OfflineZipformerCtcModelConfig;");
  jobject zipformer_ctc_config = env->GetObjectField(model_config, fid);
  jclass zipformer_ctc_config_cls = env->GetObjectClass(zipformer_ctc_config);

  fid =
      env->GetFieldID(zipformer_ctc_config_cls, "model", "Ljava/lang/String;");

  s = (jstring)env->GetObjectField(zipformer_ctc_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.zipformer_ctc.model = p;
  env->ReleaseStringUTFChars(s, p);

  // canary
  fid = env->GetFieldID(model_config_cls, "canary",
                        "Lcom/k2fsa/sherpa/onnx/OfflineCanaryModelConfig;");
  jobject canary_config = env->GetObjectField(model_config, fid);
  jclass canary_config_cls = env->GetObjectClass(canary_config);

  fid = env->GetFieldID(canary_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(canary_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.canary.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(canary_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(canary_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.canary.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(canary_config_cls, "srcLang", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(canary_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.canary.src_lang = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(canary_config_cls, "tgtLang", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(canary_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.canary.tgt_lang = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(canary_config_cls, "usePnc", "Z");
  ans.model_config.canary.use_pnc = env->GetBooleanField(canary_config, fid);

  // dolphin
  fid = env->GetFieldID(model_config_cls, "dolphin",
                        "Lcom/k2fsa/sherpa/onnx/OfflineDolphinModelConfig;");
  jobject dolphin_config = env->GetObjectField(model_config, fid);
  jclass dolphin_config_cls = env->GetObjectClass(dolphin_config);

  fid = env->GetFieldID(dolphin_config_cls, "model", "Ljava/lang/String;");

  s = (jstring)env->GetObjectField(dolphin_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.dolphin.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "teleSpeech", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.telespeech_ctc = p;
  env->ReleaseStringUTFChars(s, p);

  // homophone replacer config
  fid = env->GetFieldID(cls, "hr",
                        "Lcom/k2fsa/sherpa/onnx/HomophoneReplacerConfig;");
  jobject hr_config = env->GetObjectField(config, fid);
  jclass hr_config_cls = env->GetObjectClass(hr_config);

  fid = env->GetFieldID(hr_config_cls, "dictDir", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(hr_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.hr.dict_dir = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(hr_config_cls, "lexicon", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(hr_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.hr.lexicon = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(hr_config_cls, "ruleFsts", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(hr_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.hr.rule_fsts = p;
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
    return 0;
  }
#endif
  auto config = sherpa_onnx::GetOfflineConfig(env, _config);

  if (config.model_config.debug) {
    // logcat truncates long strings, so we split the string into chunks
    auto str_vec = sherpa_onnx::SplitString(config.ToString(), 128);
    for (const auto &s : str_vec) {
      SHERPA_ONNX_LOGE("%s", s.c_str());
    }
  }

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

  if (config.model_config.debug) {
    auto str_vec = sherpa_onnx::SplitString(config.ToString(), 128);
    for (const auto &s : str_vec) {
      SHERPA_ONNX_LOGE("%s", s.c_str());
    }
  }

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto model = new sherpa_onnx::OfflineRecognizer(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_setConfig(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jobject _config) {
  auto config = sherpa_onnx::GetOfflineConfig(env, _config);

  if (config.model_config.debug) {
    SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  }

  auto recognizer = reinterpret_cast<sherpa_onnx::OfflineRecognizer *>(ptr);
  recognizer->SetConfig(config);
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
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  SafeJNI(env, "OfflineRecognizer_decode", [&] {
    if (!ValidatePointer(env, ptr, "OfflineRecognizer_decode",
                         "OfflineRecognizer pointer is null.") ||
        !ValidatePointer(env, stream_ptr, "OfflineRecognizer_decode",
                         "OfflineStream pointer is null.")) {
      return;
    }

    auto recognizer = reinterpret_cast<sherpa_onnx::OfflineRecognizer *>(ptr);
    auto stream = reinterpret_cast<sherpa_onnx::OfflineStream *>(stream_ptr);
    recognizer->DecodeStream(stream);
  });
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_decodeStreams(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlongArray stream_ptrs) {
  SafeJNI(env, "OfflineRecognizer_decode_streams", [&] {
    if (!ValidatePointer(env, ptr, "OfflineRecognizer_decode_streams",
                         "OfflineRecognizer pointer is null.")) {
      return;
    }

    auto recognizer = reinterpret_cast<sherpa_onnx::OfflineRecognizer *>(ptr);

    jlong *p = env->GetLongArrayElements(stream_ptrs, nullptr);
    jsize n = env->GetArrayLength(stream_ptrs);

    auto ss = reinterpret_cast<sherpa_onnx::OfflineStream **>(p);
    recognizer->DecodeStreams(ss, n);

    env->ReleaseLongArrayElements(stream_ptrs, p, JNI_ABORT);
  });
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
  // [3]: lang, jstring
  // [4]: emotion, jstring
  // [5]: event, jstring
  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      6, env->FindClass("java/lang/Object"), nullptr);

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

  // [3]: lang, jstring
  // [4]: emotion, jstring
  // [5]: event, jstring
  env->SetObjectArrayElement(obj_arr, 3,
                             env->NewStringUTF(result.lang.c_str()));
  env->SetObjectArrayElement(obj_arr, 4,
                             env->NewStringUTF(result.emotion.c_str()));
  env->SetObjectArrayElement(obj_arr, 5,
                             env->NewStringUTF(result.event.c_str()));

  return obj_arr;
}
