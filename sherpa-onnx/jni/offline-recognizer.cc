// sherpa-onnx/jni/offline-recognizer.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-recognizer.h"

#include <stdlib.h>

#include <memory>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static OfflineRecognizerConfig GetOfflineConfig(JNIEnv *env, jobject config,
                                                bool *ok) {
  OfflineRecognizerConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

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

  fid = env->GetFieldID(cls, "modelConfig",
                        "Lcom/k2fsa/sherpa/onnx/OfflineModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.tokens, tokens, model_config_cls,
                              model_config);

  SHERPA_ONNX_JNI_READ_INT(ans.model_config.num_threads, numThreads,
                           model_config_cls, model_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.model_config.debug, debug, model_config_cls,
                            model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.provider, provider,
                              model_config_cls, model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.model_type, modelType,
                              model_config_cls, model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.modeling_unit, modelingUnit,
                              model_config_cls, model_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.bpe_vocab, bpeVocab,
                              model_config_cls, model_config);

  fid = env->GetFieldID(model_config_cls, "transducer",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTransducerModelConfig;");
  jobject transducer_config = env->GetObjectField(model_config, fid);
  jclass transducer_config_cls = env->GetObjectClass(transducer_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.transducer.encoder_filename,
                              encoder, transducer_config_cls,
                              transducer_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.transducer.decoder_filename,
                              decoder, transducer_config_cls,
                              transducer_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.transducer.joiner_filename,
                              joiner, transducer_config_cls, transducer_config);

  fid = env->GetFieldID(model_config_cls, "paraformer",
                        "Lcom/k2fsa/sherpa/onnx/OfflineParaformerModelConfig;");
  jobject paraformer_config = env->GetObjectField(model_config, fid);
  jclass paraformer_config_cls = env->GetObjectClass(paraformer_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.paraformer.model, model,
                              paraformer_config_cls, paraformer_config);

  fid = env->GetFieldID(paraformer_config_cls, "qnnConfig",
                        "Lcom/k2fsa/sherpa/onnx/QnnConfig;");
  jobject qnn_config = env->GetObjectField(paraformer_config, fid);
  jclass qnn_config_cls = env->GetObjectClass(qnn_config);

  SHERPA_ONNX_JNI_READ_STRING(
      ans.model_config.paraformer.qnn_config.backend_lib, backendLib,
      qnn_config_cls, qnn_config);

  SHERPA_ONNX_JNI_READ_STRING(
      ans.model_config.paraformer.qnn_config.context_binary, contextBinary,
      qnn_config_cls, qnn_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.paraformer.qnn_config.system_lib,
                              systemLib, qnn_config_cls, qnn_config);

  fid = env->GetFieldID(model_config_cls, "whisper",
                        "Lcom/k2fsa/sherpa/onnx/OfflineWhisperModelConfig;");
  jobject whisper_config = env->GetObjectField(model_config, fid);
  jclass whisper_config_cls = env->GetObjectClass(whisper_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.whisper.encoder, encoder,
                              whisper_config_cls, whisper_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.whisper.decoder, decoder,
                              whisper_config_cls, whisper_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.whisper.language, language,
                              whisper_config_cls, whisper_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.whisper.task, task,
                              whisper_config_cls, whisper_config);

  SHERPA_ONNX_JNI_READ_INT(ans.model_config.whisper.tail_paddings, tailPaddings,
                           whisper_config_cls, whisper_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.model_config.whisper.enable_token_timestamps,
                            enableTokenTimestamps, whisper_config_cls,
                            whisper_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.model_config.whisper.enable_segment_timestamps,
                            enableSegmentTimestamps, whisper_config_cls,
                            whisper_config);

  fid = env->GetFieldID(model_config_cls, "fireRedAsr",
                        "Lcom/k2fsa/sherpa/onnx/OfflineFireRedAsrModelConfig;");
  jobject fire_red_asr_config = env->GetObjectField(model_config, fid);
  jclass fire_red_asr_config_cls = env->GetObjectClass(fire_red_asr_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.fire_red_asr.encoder, encoder,
                              fire_red_asr_config_cls, fire_red_asr_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.fire_red_asr.decoder, decoder,
                              fire_red_asr_config_cls, fire_red_asr_config);

  // moonshine
  fid = env->GetFieldID(model_config_cls, "moonshine",
                        "Lcom/k2fsa/sherpa/onnx/OfflineMoonshineModelConfig;");
  jobject moonshine_config = env->GetObjectField(model_config, fid);
  jclass moonshine_config_cls = env->GetObjectClass(moonshine_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.moonshine.preprocessor,
                              preprocessor, moonshine_config_cls,
                              moonshine_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.moonshine.encoder, encoder,
                              moonshine_config_cls, moonshine_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.moonshine.uncached_decoder,
                              uncachedDecoder, moonshine_config_cls,
                              moonshine_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.moonshine.cached_decoder,
                              cachedDecoder, moonshine_config_cls,
                              moonshine_config);

  fid = env->GetFieldID(model_config_cls, "senseVoice",
                        "Lcom/k2fsa/sherpa/onnx/OfflineSenseVoiceModelConfig;");
  jobject sense_voice_config = env->GetObjectField(model_config, fid);
  jclass sense_voice_config_cls = env->GetObjectClass(sense_voice_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.sense_voice.model, model,
                              sense_voice_config_cls, sense_voice_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.sense_voice.language, language,
                              sense_voice_config_cls, sense_voice_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.model_config.sense_voice.use_itn,
                            useInverseTextNormalization, sense_voice_config_cls,
                            sense_voice_config);

  fid = env->GetFieldID(sense_voice_config_cls, "qnnConfig",
                        "Lcom/k2fsa/sherpa/onnx/QnnConfig;");
  qnn_config = env->GetObjectField(sense_voice_config, fid);
  qnn_config_cls = env->GetObjectClass(qnn_config);

  SHERPA_ONNX_JNI_READ_STRING(
      ans.model_config.sense_voice.qnn_config.backend_lib, backendLib,
      qnn_config_cls, qnn_config);

  SHERPA_ONNX_JNI_READ_STRING(
      ans.model_config.sense_voice.qnn_config.context_binary, contextBinary,
      qnn_config_cls, qnn_config);

  SHERPA_ONNX_JNI_READ_STRING(
      ans.model_config.sense_voice.qnn_config.system_lib, systemLib,
      qnn_config_cls, qnn_config);

  // nemo
  fid = env->GetFieldID(
      model_config_cls, "nemo",
      "Lcom/k2fsa/sherpa/onnx/OfflineNemoEncDecCtcModelConfig;");
  jobject nemo_config = env->GetObjectField(model_config, fid);
  jclass nemo_config_cls = env->GetObjectClass(nemo_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.nemo_ctc.model, model,
                              nemo_config_cls, nemo_config);

  // zipformer ctc
  fid =
      env->GetFieldID(model_config_cls, "zipformerCtc",
                      "Lcom/k2fsa/sherpa/onnx/OfflineZipformerCtcModelConfig;");
  jobject zipformer_ctc_config = env->GetObjectField(model_config, fid);
  jclass zipformer_ctc_config_cls = env->GetObjectClass(zipformer_ctc_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.zipformer_ctc.model, model,
                              zipformer_ctc_config_cls, zipformer_ctc_config);

  fid = env->GetFieldID(zipformer_ctc_config_cls, "qnnConfig",
                        "Lcom/k2fsa/sherpa/onnx/QnnConfig;");

  qnn_config = env->GetObjectField(zipformer_ctc_config, fid);
  qnn_config_cls = env->GetObjectClass(qnn_config);

  SHERPA_ONNX_JNI_READ_STRING(
      ans.model_config.zipformer_ctc.qnn_config.backend_lib, backendLib,
      qnn_config_cls, qnn_config);

  SHERPA_ONNX_JNI_READ_STRING(
      ans.model_config.zipformer_ctc.qnn_config.context_binary, contextBinary,
      qnn_config_cls, qnn_config);

  SHERPA_ONNX_JNI_READ_STRING(
      ans.model_config.zipformer_ctc.qnn_config.system_lib, systemLib,
      qnn_config_cls, qnn_config);

  // wenet ctc
  fid = env->GetFieldID(model_config_cls, "wenetCtc",
                        "Lcom/k2fsa/sherpa/onnx/OfflineWenetCtcModelConfig;");
  jobject wenet_ctc_config = env->GetObjectField(model_config, fid);
  jclass wenet_ctc_config_cls = env->GetObjectClass(wenet_ctc_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.wenet_ctc.model, model,
                              wenet_ctc_config_cls, wenet_ctc_config);

  // omnilingual asr ctc
  fid = env->GetFieldID(
      model_config_cls, "omnilingual",
      "Lcom/k2fsa/sherpa/onnx/OfflineOmnilingualAsrCtcModelConfig;");
  jobject omnilingual_ctc_config = env->GetObjectField(model_config, fid);
  jclass omnilingual_ctc_config_cls =
      env->GetObjectClass(omnilingual_ctc_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.omnilingual.model, model,
                              omnilingual_ctc_config_cls,
                              omnilingual_ctc_config);

  // medasr ctc
  fid = env->GetFieldID(model_config_cls, "medasr",
                        "Lcom/k2fsa/sherpa/onnx/OfflineMedAsrCtcModelConfig;");
  jobject medasr_ctc_config = env->GetObjectField(model_config, fid);
  jclass medasr_ctc_config_cls = env->GetObjectClass(medasr_ctc_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.medasr.model, model,
                              medasr_ctc_config_cls, medasr_ctc_config);

  // FunASR Nano
  fid = env->GetFieldID(model_config_cls, "funasrNano",
                        "Lcom/k2fsa/sherpa/onnx/OfflineFunAsrNanoModelConfig;");
  jobject funasr_nano_config = env->GetObjectField(model_config, fid);
  jclass funasr_nano_config_cls = env->GetObjectClass(funasr_nano_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.funasr_nano.encoder_adaptor,
                              encoderAdaptor, funasr_nano_config_cls,
                              funasr_nano_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.funasr_nano.llm, llm,
                              funasr_nano_config_cls, funasr_nano_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.funasr_nano.embedding, embedding,
                              funasr_nano_config_cls, funasr_nano_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.funasr_nano.tokenizer, tokenizer,
                              funasr_nano_config_cls, funasr_nano_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.funasr_nano.system_prompt,
                              systemPrompt, funasr_nano_config_cls,
                              funasr_nano_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.funasr_nano.user_prompt,
                              userPrompt, funasr_nano_config_cls,
                              funasr_nano_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.funasr_nano.language, language,
                              funasr_nano_config_cls, funasr_nano_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.model_config.funasr_nano.itn, itn,
                            funasr_nano_config_cls, funasr_nano_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.funasr_nano.hotwords, hotwords,
                              funasr_nano_config_cls, funasr_nano_config);

  SHERPA_ONNX_JNI_READ_INT(ans.model_config.funasr_nano.max_new_tokens,
                           maxNewTokens, funasr_nano_config_cls,
                           funasr_nano_config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model_config.funasr_nano.temperature,
                             temperature, funasr_nano_config_cls,
                             funasr_nano_config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model_config.funasr_nano.top_p, topP,
                             funasr_nano_config_cls, funasr_nano_config);

  SHERPA_ONNX_JNI_READ_INT(ans.model_config.funasr_nano.seed, seed,
                           funasr_nano_config_cls, funasr_nano_config);

  // fire red asr ctc
  fid = env->GetFieldID(
      model_config_cls, "fireRedAsrCtc",
      "Lcom/k2fsa/sherpa/onnx/OfflineFireRedAsrCtcModelConfig;");
  jobject fire_red_asr_ctc_config = env->GetObjectField(model_config, fid);
  jclass fire_red_asr_ctc_config_cls =
      env->GetObjectClass(fire_red_asr_ctc_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.fire_red_asr_ctc.model, model,
                              fire_red_asr_ctc_config_cls,
                              fire_red_asr_ctc_config);

  // canary
  fid = env->GetFieldID(model_config_cls, "canary",
                        "Lcom/k2fsa/sherpa/onnx/OfflineCanaryModelConfig;");
  jobject canary_config = env->GetObjectField(model_config, fid);
  jclass canary_config_cls = env->GetObjectClass(canary_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.canary.encoder, encoder,
                              canary_config_cls, canary_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.canary.decoder, decoder,
                              canary_config_cls, canary_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.canary.src_lang, srcLang,
                              canary_config_cls, canary_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.canary.tgt_lang, tgtLang,
                              canary_config_cls, canary_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.model_config.canary.use_pnc, usePnc,
                            canary_config_cls, canary_config);

  fid = env->GetFieldID(model_config_cls, "dolphin",
                        "Lcom/k2fsa/sherpa/onnx/OfflineDolphinModelConfig;");
  jobject dolphin_config = env->GetObjectField(model_config, fid);
  jclass dolphin_config_cls = env->GetObjectClass(dolphin_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.dolphin.model, model,
                              dolphin_config_cls, dolphin_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_config.telespeech_ctc, teleSpeech,
                              model_config_cls, model_config);

  // homophone replacer config
  fid = env->GetFieldID(cls, "hr",
                        "Lcom/k2fsa/sherpa/onnx/HomophoneReplacerConfig;");
  jobject hr_config = env->GetObjectField(config, fid);
  jclass hr_config_cls = env->GetObjectClass(hr_config);

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
  bool ok = false;
  auto config = sherpa_onnx::GetOfflineConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  if (config.model_config.debug) {
#if __ANDROID_API__
    // logcat truncates long strings, so we split the string into chunks
    auto str_vec = sherpa_onnx::SplitString(config.ToString(), 128);
    for (const auto &s : str_vec) {
      SHERPA_ONNX_LOGE("%s", s.c_str());
    }
#else
    SHERPA_ONNX_LOGE("%s", config.ToString().c_str());
#endif
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
  bool ok = false;
  auto config = sherpa_onnx::GetOfflineConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  if (config.model_config.debug) {
#if __ANDROID_API__
    auto str_vec = sherpa_onnx::SplitString(config.ToString(), 128);
    for (const auto &s : str_vec) {
      SHERPA_ONNX_LOGE("%s", s.c_str());
    }
#else
    SHERPA_ONNX_LOGE("%s", config.ToString().c_str());
#endif
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
  bool ok = false;
  auto config = sherpa_onnx::GetOfflineConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return;
  }

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
JNIEXPORT jobject JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_getResult(JNIEnv *env,
                                                       jobject /*obj*/,
                                                       jlong streamPtr) {
  auto stream = reinterpret_cast<sherpa_onnx::OfflineStream *>(streamPtr);
  sherpa_onnx::OfflineRecognitionResult result = stream->GetResult();

  // 2. Find the Java class and constructor
  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/OfflineRecognizerResult");
  jmethodID ctor =
      env->GetMethodID(cls, "<init>",
                       "(Ljava/lang/String;[Ljava/lang/String;[FLjava/lang/"
                       "String;Ljava/lang/String;Ljava/lang/String;[F)V");
  jstring jtext = env->NewStringUTF(result.text.c_str());

  jobjectArray jtokens = env->NewObjectArray(
      result.tokens.size(), env->FindClass("java/lang/String"), nullptr);

  for (size_t i = 0; i < result.tokens.size(); ++i) {
    jstring token_str = env->NewStringUTF(result.tokens[i].c_str());
    env->SetObjectArrayElement(jtokens, i, token_str);
    env->DeleteLocalRef(token_str);
  }

  jfloatArray jtimestamps = env->NewFloatArray(result.timestamps.size());
  env->SetFloatArrayRegion(jtimestamps, 0, result.timestamps.size(),
                           result.timestamps.data());

  jstring jlang = env->NewStringUTF(result.lang.c_str());
  jstring jemotion = env->NewStringUTF(result.emotion.c_str());
  jstring jevent = env->NewStringUTF(result.event.c_str());

  jfloatArray jdurations = env->NewFloatArray(result.durations.size());
  env->SetFloatArrayRegion(jdurations, 0, result.durations.size(),
                           result.durations.data());

  jobject jresult = env->NewObject(cls, ctor, jtext, jtokens, jtimestamps,
                                   jlang, jemotion, jevent, jdurations);

  env->DeleteLocalRef(jtext);
  env->DeleteLocalRef(jtokens);
  env->DeleteLocalRef(jtimestamps);
  env->DeleteLocalRef(jlang);
  env->DeleteLocalRef(jemotion);
  env->DeleteLocalRef(jevent);
  env->DeleteLocalRef(jdurations);
  env->DeleteLocalRef(cls);

  return jresult;  // returned object is safe
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineRecognizer_prependAdspLibraryPath(
    JNIEnv *env, jclass /*cls*/, jstring new_path) {
  const char *p = env->GetStringUTFChars(new_path, nullptr);
  sherpa_onnx::PrependAdspLibraryPath(p);

  env->ReleaseStringUTFChars(new_path, p);
}
