// sherpa-onnx/jni/offline-tts.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/wave-writer.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static GenerationConfig GetGenerationConfig(JNIEnv *env, jobject config_obj) {
  GenerationConfig ans;

  if (config_obj == nullptr) {
    SHERPA_ONNX_LOGE("GenerationConfig is null");
    return ans;
  }

  jclass cls = env->GetObjectClass(config_obj);
  jfieldID fid;

  // silenceScale
  fid = env->GetFieldID(cls, "silenceScale", "F");
  if (fid != nullptr) ans.silence_scale = env->GetFloatField(config_obj, fid);

  // speed
  fid = env->GetFieldID(cls, "speed", "F");
  if (fid != nullptr) ans.speed = env->GetFloatField(config_obj, fid);

  // sid
  fid = env->GetFieldID(cls, "sid", "I");
  if (fid != nullptr) ans.sid = env->GetIntField(config_obj, fid);

  // referenceAudio
  fid = env->GetFieldID(cls, "referenceAudio", "[F");
  if (fid != nullptr) {
    jfloatArray arr = (jfloatArray)env->GetObjectField(config_obj, fid);
    if (arr != nullptr) {
      jsize len = env->GetArrayLength(arr);
      jfloat *elems = env->GetFloatArrayElements(arr, nullptr);
      ans.reference_audio.assign(elems, elems + len);
      env->ReleaseFloatArrayElements(arr, elems, JNI_ABORT);
    }
  }

  // referenceSampleRate
  fid = env->GetFieldID(cls, "referenceSampleRate", "I");
  if (fid != nullptr)
    ans.reference_sample_rate = env->GetIntField(config_obj, fid);

  // referenceText
  fid = env->GetFieldID(cls, "referenceText", "Ljava/lang/String;");
  if (fid != nullptr) {
    jstring str = (jstring)env->GetObjectField(config_obj, fid);
    if (str != nullptr) {
      const char *chars = env->GetStringUTFChars(str, nullptr);
      ans.reference_text = chars;
      env->ReleaseStringUTFChars(str, chars);
    }
  }

  // numSteps
  fid = env->GetFieldID(cls, "numSteps", "I");
  if (fid != nullptr) ans.num_steps = env->GetIntField(config_obj, fid);

  // extra Map<String, String>
  fid = env->GetFieldID(cls, "extra", "Ljava/util/Map;");
  if (fid != nullptr) {
    jobject map_obj = env->GetObjectField(config_obj, fid);
    if (map_obj != nullptr) {
      jclass map_cls = env->GetObjectClass(map_obj);
      jmethodID entrySet =
          env->GetMethodID(map_cls, "entrySet", "()Ljava/util/Set;");
      jobject entry_set = env->CallObjectMethod(map_obj, entrySet);

      jclass set_cls = env->GetObjectClass(entry_set);
      jmethodID iteratorMid =
          env->GetMethodID(set_cls, "iterator", "()Ljava/util/Iterator;");
      jobject iterator = env->CallObjectMethod(entry_set, iteratorMid);

      jclass iter_cls = env->GetObjectClass(iterator);
      jmethodID hasNextMid = env->GetMethodID(iter_cls, "hasNext", "()Z");
      jmethodID nextMid =
          env->GetMethodID(iter_cls, "next", "()Ljava/lang/Object;");

      jclass entry_cls = env->FindClass("java/util/Map$Entry");
      jmethodID getKeyMid =
          env->GetMethodID(entry_cls, "getKey", "()Ljava/lang/Object;");
      jmethodID getValueMid =
          env->GetMethodID(entry_cls, "getValue", "()Ljava/lang/Object;");

      while (env->CallBooleanMethod(iterator, hasNextMid)) {
        jobject entry = env->CallObjectMethod(iterator, nextMid);
        jstring key = (jstring)env->CallObjectMethod(entry, getKeyMid);
        jstring value = (jstring)env->CallObjectMethod(entry, getValueMid);

        const char *keyChars = env->GetStringUTFChars(key, nullptr);
        const char *valueChars = env->GetStringUTFChars(value, nullptr);

        ans.extra[std::string(keyChars)] = std::string(valueChars);

        env->ReleaseStringUTFChars(key, keyChars);
        env->ReleaseStringUTFChars(value, valueChars);
      }
    }
  }

  return ans;
}

static OfflineTtsConfig GetOfflineTtsConfig(JNIEnv *env, jobject config,
                                            bool *ok) {
  OfflineTtsConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  fid = env->GetFieldID(cls, "model",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTtsModelConfig;");
  jobject model = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model);

  fid = env->GetFieldID(model_config_cls, "vits",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTtsVitsModelConfig;");
  jobject vits = env->GetObjectField(model, fid);
  jclass vits_cls = env->GetObjectClass(vits);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.vits.model, model, vits_cls, vits);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.vits.lexicon, lexicon, vits_cls, vits);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.vits.tokens, tokens, vits_cls, vits);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.vits.data_dir, dataDir, vits_cls, vits);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.vits.noise_scale, noiseScale, vits_cls,
                             vits);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.vits.noise_scale_w, noiseScaleW,
                             vits_cls, vits);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.vits.length_scale, lengthScale, vits_cls,
                             vits);

  // matcha
  fid = env->GetFieldID(model_config_cls, "matcha",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTtsMatchaModelConfig;");
  jobject matcha = env->GetObjectField(model, fid);
  jclass matcha_cls = env->GetObjectClass(matcha);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.matcha.acoustic_model, acousticModel,
                              matcha_cls, matcha);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.matcha.vocoder, vocoder, matcha_cls,
                              matcha);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.matcha.lexicon, lexicon, matcha_cls,
                              matcha);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.matcha.tokens, tokens, matcha_cls,
                              matcha);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.matcha.data_dir, dataDir, matcha_cls,
                              matcha);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.matcha.noise_scale, noiseScale,
                             matcha_cls, matcha);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.matcha.length_scale, lengthScale,
                             matcha_cls, matcha);

  fid = env->GetFieldID(model_config_cls, "kokoro",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTtsKokoroModelConfig;");
  jobject kokoro = env->GetObjectField(model, fid);
  jclass kokoro_cls = env->GetObjectClass(kokoro);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kokoro.model, model, kokoro_cls,
                              kokoro);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kokoro.voices, voices, kokoro_cls,
                              kokoro);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kokoro.tokens, tokens, kokoro_cls,
                              kokoro);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kokoro.lexicon, lexicon, kokoro_cls,
                              kokoro);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kokoro.lang, lang, kokoro_cls, kokoro);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kokoro.data_dir, dataDir, kokoro_cls,
                              kokoro);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.kokoro.length_scale, lengthScale,
                             kokoro_cls, kokoro);

  // kitten
  fid = env->GetFieldID(model_config_cls, "kitten",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTtsKittenModelConfig;");
  jobject kitten = env->GetObjectField(model, fid);
  jclass kitten_cls = env->GetObjectClass(kitten);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kitten.model, model, kitten_cls,
                              kitten);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kitten.voices, voices, kitten_cls,
                              kitten);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kitten.tokens, tokens, kitten_cls,
                              kitten);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.kitten.data_dir, dataDir, kitten_cls,
                              kitten);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.kitten.length_scale, lengthScale,
                             kitten_cls, kitten);

  // pocket
  fid = env->GetFieldID(model_config_cls, "pocket",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTtsPocketModelConfig;");
  jobject pocket = env->GetObjectField(model, fid);
  jclass pocket_cls = env->GetObjectClass(pocket);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.pocket.lm_flow, lmFlow, pocket_cls,
                              pocket);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.pocket.lm_main, lmMain, pocket_cls,
                              pocket);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.pocket.encoder, encoder, pocket_cls,
                              pocket);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.pocket.decoder, decoder, pocket_cls,
                              pocket);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.pocket.text_conditioner,
                              textConditioner, pocket_cls, pocket);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.pocket.vocab_json, vocabJson,
                              pocket_cls, pocket);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.pocket.token_scores_json,
                              tokenScoresJson, pocket_cls, pocket);

  SHERPA_ONNX_JNI_READ_INT(ans.model.num_threads, numThreads, model_config_cls,
                           model);

  SHERPA_ONNX_JNI_READ_BOOL(ans.model.debug, debug, model_config_cls, model);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.provider, provider, model_config_cls,
                              model);

  SHERPA_ONNX_JNI_READ_STRING(ans.rule_fsts, ruleFsts, cls, config);

  SHERPA_ONNX_JNI_READ_STRING(ans.rule_fars, ruleFars, cls, config);

  SHERPA_ONNX_JNI_READ_INT(ans.max_num_sentences, maxNumSentences, cls, config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.silence_scale, silenceScale, cls, config);

  *ok = true;
  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif

  bool ok = false;
  auto config = sherpa_onnx::GetOfflineTtsConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  auto str_vec = sherpa_onnx::SplitString(config.ToString(), 128);
  for (const auto &s : str_vec) {
    SHERPA_ONNX_LOGE("%s", s.c_str());
  }

  auto tts = new sherpa_onnx::OfflineTts(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)tts;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  return SafeJNI(
      env, "OfflineTts_newFromFile",
      [&]() -> jlong {
        bool ok = false;
        auto config = sherpa_onnx::GetOfflineTtsConfig(env, _config, &ok);

        if (!ok) {
          SHERPA_ONNX_LOGE("Please read the error message carefully");
          return 0;
        }

        SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

        if (!config.Validate()) {
          SHERPA_ONNX_LOGE("Errors found in config!");
        }

        auto tts = new sherpa_onnx::OfflineTts(config);
        return reinterpret_cast<jlong>(tts);
      },
      (jlong)0);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_getSampleRate(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr)->SampleRate();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_getNumSpeakers(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr)->NumSpeakers();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineTts_generateImpl(JNIEnv *env, jobject /*obj*/,
                                                   jlong ptr, jstring text,
                                                   jint sid, jfloat speed) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);

  auto audio = reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr)->Generate(
      p_text, sid, speed);

  jfloatArray samples_arr = env->NewFloatArray(audio.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, audio.samples.size(),
                           audio.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, audio.sample_rate));

  env->ReleaseStringUTFChars(text, p_text);

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineTts_generateWithCallbackImpl(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring text, jint sid,
    jfloat speed, jobject callback) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);

  std::function<int32_t(const float *, int32_t, float)> callback_wrapper =
      [env, callback](const float *samples, int32_t n,
                      float /*progress*/) -> int {
    if (!callback) return 1;

    jfloatArray samples_arr = env->NewFloatArray(n);
    env->SetFloatArrayRegion(samples_arr, 0, n, samples);

    jclass cls = env->GetObjectClass(callback);

    // Try Kotlin-style lambda first (boxed Int)
    jmethodID mid = env->GetMethodID(cls, "invoke", "([F)Ljava/lang/Integer;");
    if (mid != nullptr) {
      jobject result = env->CallObjectMethod(callback, mid, samples_arr);
      if (result != nullptr) {
        jclass int_cls = env->GetObjectClass(result);
        jmethodID int_value_mid = env->GetMethodID(int_cls, "intValue", "()I");
        return env->CallIntMethod(result, int_value_mid);
      }
      return 1;
    }

    // Optional: Java primitive int apply([F)I fallback for pure Java callbacks
    mid = env->GetMethodID(cls, "apply", "([F)I");
    if (mid != nullptr) {
      return env->CallIntMethod(callback, mid, samples_arr);
    }

    // Optional: Java boxed Integer apply([F)Integer fallback
    mid = env->GetMethodID(cls, "apply", "([F)Ljava/lang/Integer;");
    if (mid != nullptr) {
      jobject result = env->CallObjectMethod(callback, mid, samples_arr);
      if (result != nullptr) {
        jclass int_cls = env->GetObjectClass(result);
        jmethodID int_value_mid = env->GetMethodID(int_cls, "intValue", "()I");
        return env->CallIntMethod(result, int_value_mid);
      }
      return 1;
    }

    return 1;  // default
  };

  auto tts = reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr);
  auto audio = tts->Generate(p_text, sid, speed, callback_wrapper);

  jfloatArray samples_arr = env->NewFloatArray(audio.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, audio.samples.size(),
                           audio.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, audio.sample_rate));

  env->ReleaseStringUTFChars(text, p_text);

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineTts_generateWithConfigImpl(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring text, jobject _gen_config,
    jobject callback) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);
  auto gen_config = sherpa_onnx::GetGenerationConfig(env, _gen_config);

  std::function<int32_t(const float *, int32_t, float)> callback_wrapper =
      [env, callback](const float *samples, int32_t n,
                      float /*progress*/) -> int {
    if (!callback) return 1;

    jfloatArray samples_arr = env->NewFloatArray(n);
    env->SetFloatArrayRegion(samples_arr, 0, n, samples);

    jclass cls = env->GetObjectClass(callback);

    // Try Kotlin-style lambda first (boxed Int)
    jmethodID mid = env->GetMethodID(cls, "invoke", "([F)Ljava/lang/Integer;");
    if (mid != nullptr) {
      jobject result = env->CallObjectMethod(callback, mid, samples_arr);
      if (result != nullptr) {
        jclass int_cls = env->GetObjectClass(result);
        jmethodID int_value_mid = env->GetMethodID(int_cls, "intValue", "()I");
        return env->CallIntMethod(result, int_value_mid);
      }
      return 1;
    }

    // Optional: Java primitive int apply([F)I fallback for pure Java callbacks
    mid = env->GetMethodID(cls, "apply", "([F)I");
    if (mid != nullptr) {
      return env->CallIntMethod(callback, mid, samples_arr);
    }

    // Optional: Java boxed Integer apply([F)Integer fallback
    mid = env->GetMethodID(cls, "apply", "([F)Ljava/lang/Integer;");
    if (mid != nullptr) {
      jobject result = env->CallObjectMethod(callback, mid, samples_arr);
      if (result != nullptr) {
        jclass int_cls = env->GetObjectClass(result);
        jmethodID int_value_mid = env->GetMethodID(int_cls, "intValue", "()I");
        return env->CallIntMethod(result, int_value_mid);
      }
      return 1;
    }

    return 1;  // default
  };

  auto tts = reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr);
  auto audio = tts->Generate(p_text, gen_config, callback_wrapper);

  // Convert to Java array
  jfloatArray samples_arr = env->NewFloatArray(audio.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, audio.samples.size(),
                           audio.samples.data());

  jobjectArray obj_arr =
      env->NewObjectArray(2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, audio.sample_rate));

  env->ReleaseStringUTFChars(text, p_text);
  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL Java_com_k2fsa_sherpa_onnx_GeneratedAudio_saveImpl(
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
