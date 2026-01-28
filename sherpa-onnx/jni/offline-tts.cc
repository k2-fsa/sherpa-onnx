// sherpa-onnx/jni/offline-tts.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/wave-writer.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

// ------------------ JNI Config Helpers ------------------

static GenerationConfig GetGenerationConfig(JNIEnv *env, jobject config_obj) {
  GenerationConfig ans;

  if (!config_obj) {
    SHERPA_ONNX_LOGE("GenerationConfig is null");
    return ans;
  }

  jclass cls = env->GetObjectClass(config_obj);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.silence_scale, silenceScale, cls, config_obj);
  SHERPA_ONNX_JNI_READ_FLOAT(ans.speed, speed, cls, config_obj);
  SHERPA_ONNX_JNI_READ_INT(ans.sid, sid, cls, config_obj);

  // referenceAudio
  jfieldID fid = env->GetFieldID(cls, "referenceAudio", "[F");
  if (fid != nullptr) {
    jfloatArray arr = (jfloatArray)env->GetObjectField(config_obj, fid);
    if (arr != nullptr) {
      jsize len = env->GetArrayLength(arr);
      jfloat *elems = env->GetFloatArrayElements(arr, nullptr);
      ans.reference_audio.assign(elems, elems + len);
      env->ReleaseFloatArrayElements(arr, elems, JNI_ABORT);
      env->DeleteLocalRef(arr);
    }
  }

  SHERPA_ONNX_JNI_READ_INT(ans.reference_sample_rate, referenceSampleRate, cls,
                           config_obj);

  // referenceText
  SHERPA_ONNX_JNI_READ_STRING(ans.reference_text, referenceText, cls,
                              config_obj);

  SHERPA_ONNX_JNI_READ_INT(ans.num_steps, numSteps, cls, config_obj);

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
        if (!entry) {
          continue;
        }

        jstring key = (jstring)env->CallObjectMethod(entry, getKeyMid);
        jstring value = (jstring)env->CallObjectMethod(entry, getValueMid);

        if (key != nullptr && value != nullptr) {
          const char *keyChars = env->GetStringUTFChars(key, nullptr);
          const char *valueChars = env->GetStringUTFChars(value, nullptr);
          ans.extra[std::string(keyChars)] = std::string(valueChars);

          env->ReleaseStringUTFChars(key, keyChars);
          env->ReleaseStringUTFChars(value, valueChars);
        }

        env->DeleteLocalRef(key);
        env->DeleteLocalRef(value);
        env->DeleteLocalRef(entry);
      }

      env->DeleteLocalRef(entry_set);
      env->DeleteLocalRef(iterator);
      env->DeleteLocalRef(entry_cls);
      env->DeleteLocalRef(iter_cls);
      env->DeleteLocalRef(set_cls);
      env->DeleteLocalRef(map_cls);
      env->DeleteLocalRef(map_obj);
    }
  }

  env->DeleteLocalRef(cls);
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

  env->DeleteLocalRef(model);
  env->DeleteLocalRef(vits);
  env->DeleteLocalRef(vits_cls);
  env->DeleteLocalRef(matcha);
  env->DeleteLocalRef(matcha_cls);
  env->DeleteLocalRef(kokoro);
  env->DeleteLocalRef(kokoro_cls);
  env->DeleteLocalRef(kitten);
  env->DeleteLocalRef(kitten_cls);
  env->DeleteLocalRef(pocket);
  env->DeleteLocalRef(pocket_cls);
  env->DeleteLocalRef(model_config_cls);
  env->DeleteLocalRef(cls);

  *ok = true;
  return ans;
}

}  // namespace sherpa_onnx

// Convert audio samples and sample rate to a Java GeneratedAudio object
static jobject CreateAudioObject(JNIEnv *env, const std::vector<float> &samples,
                                 int32_t sample_rate) {
  // Step 1: Create a jfloatArray for samples
  jfloatArray samples_arr = env->NewFloatArray(samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, samples.size(), samples.data());

  // Step 2: Find the GeneratedAudio class
  jclass gen_audio_cls = env->FindClass("com/k2fsa/sherpa/onnx/GeneratedAudio");
  if (!gen_audio_cls) {
    env->DeleteLocalRef(samples_arr);
    return nullptr;
  }

  // Step 3: Get the constructor: GeneratedAudio(float[] samples, int
  // sampleRate)
  jmethodID ctor = env->GetMethodID(gen_audio_cls, "<init>", "([FI)V");
  if (!ctor) {
    env->DeleteLocalRef(samples_arr);
    env->DeleteLocalRef(gen_audio_cls);
    return nullptr;
  }

  // Step 4: Create the object
  jobject gen_audio_obj =
      env->NewObject(gen_audio_cls, ctor, samples_arr, sample_rate);

  // Step 5: Clean up local refs
  env->DeleteLocalRef(samples_arr);
  env->DeleteLocalRef(gen_audio_cls);

  return gen_audio_obj;
}

// ----------------- Consumer<float[]> -----------------
static int32_t CallConsumerCallback(JNIEnv *env, jobject callback,
                                    jfloatArray samples_arr) {
  jclass consumer_cls = env->FindClass("java/util/function/Consumer");
  if (env->ExceptionCheck() || !env->IsInstanceOf(callback, consumer_cls)) {
    env->DeleteLocalRef(consumer_cls);
    return -1;  // not a Consumer
  }

  jmethodID accept_mid =
      env->GetMethodID(consumer_cls, "accept", "(Ljava/lang/Object;)V");
  if (env->ExceptionCheck()) {
    env->DeleteLocalRef(consumer_cls);
    return -1;
  }

  env->CallVoidMethod(callback, accept_mid, samples_arr);
  if (env->ExceptionCheck()) {
    env->DeleteLocalRef(consumer_cls);
    return 1;  // exception occurred, continue
  }

  env->DeleteLocalRef(consumer_cls);
  return 1;  // continue
}

// ----------------- Function<float[], Integer> -----------------
static int32_t CallFunctionCallback(JNIEnv *env, jobject callback,
                                    jfloatArray samples_arr) {
  jclass function_cls = env->FindClass("java/util/function/Function");
  if (env->ExceptionCheck() || !env->IsInstanceOf(callback, function_cls)) {
    env->DeleteLocalRef(function_cls);
    return -1;  // not a Function
  }

  jmethodID apply_mid = env->GetMethodID(
      function_cls, "apply", "(Ljava/lang/Object;)Ljava/lang/Object;");
  if (env->ExceptionCheck()) {
    env->DeleteLocalRef(function_cls);
    return -1;
  }

  jobject result = env->CallObjectMethod(callback, apply_mid, samples_arr);
  if (env->ExceptionCheck() || !result) {
    env->DeleteLocalRef(function_cls);
    return 1;  // exception or null → continue
  }

  jclass integer_cls = env->FindClass("java/lang/Integer");
  jmethodID int_val_mid = env->GetMethodID(integer_cls, "intValue", "()I");
  jint ret = env->CallIntMethod(result, int_val_mid);

  env->DeleteLocalRef(integer_cls);
  env->DeleteLocalRef(result);
  env->DeleteLocalRef(function_cls);

  return ret;
}

// ----------------- OfflineTtsCallback.invoke -----------------
static int32_t CallInvokeCallback(JNIEnv *env, jobject callback,
                                  jfloatArray samples_arr) {
  jclass cls = env->GetObjectClass(callback);
  if (env->ExceptionCheck()) {
    env->DeleteLocalRef(cls);
    return 1;
  }

  jmethodID invoke_mid =
      env->GetMethodID(cls, "invoke", "([F)Ljava/lang/Integer;");
  if (env->ExceptionCheck() || !invoke_mid) {
    env->DeleteLocalRef(cls);
    return 1;
  }

  jobject result = env->CallObjectMethod(callback, invoke_mid, samples_arr);
  if (env->ExceptionCheck() || !result) {
    env->DeleteLocalRef(cls);
    return 1;
  }

  jclass integer_cls = env->GetObjectClass(result);
  jmethodID int_val_mid = env->GetMethodID(integer_cls, "intValue", "()I");
  jint ret = env->CallIntMethod(result, int_val_mid);

  env->DeleteLocalRef(integer_cls);
  env->DeleteLocalRef(result);
  env->DeleteLocalRef(cls);

  return ret;
}

static int32_t CallCallback(JNIEnv *env, jobject callback,
                            jfloatArray samples_arr) {
  if (!callback) return 1;

  int32_t ret;

  // Try Consumer
  ret = CallConsumerCallback(env, callback, samples_arr);
  if (ret != -1) return ret;

  // Try Function
  ret = CallFunctionCallback(env, callback, samples_arr);
  if (ret != -1) return ret;

  // Fallback to invoke()
  return CallInvokeCallback(env, callback, samples_arr);
}

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

  return reinterpret_cast<jlong>(tts);
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
JNIEXPORT jobject JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_generateImpl(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring text, jint sid,
    jfloat speed) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);

  auto audio = reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr)->Generate(
      p_text, sid, speed);

  env->ReleaseStringUTFChars(text, p_text);

  return CreateAudioObject(env, audio.samples, audio.sample_rate);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineTts_generateWithCallbackImpl(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring text, jint sid,
    jfloat speed, jobject callback) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);

  auto tts = reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr);

  std::function<int32_t(const float *, int32_t, float)> callback_wrapper =
      [env, callback](const float *samples, int32_t n, float) -> int32_t {
    jfloatArray samples_arr = env->NewFloatArray(n);
    env->SetFloatArrayRegion(samples_arr, 0, n, samples);
    int32_t ret = CallCallback(env, callback, samples_arr);
    env->DeleteLocalRef(samples_arr);
    return ret;
  };

  auto audio = tts->Generate(p_text, sid, speed, callback_wrapper);
  env->ReleaseStringUTFChars(text, p_text);

  return CreateAudioObject(env, audio.samples, audio.sample_rate);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineTts_generateWithConfigImpl(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring text, jobject _gen_config,
    jobject callback) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);
  auto gen_config = sherpa_onnx::GetGenerationConfig(env, _gen_config);
  auto tts = reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr);

  std::function<int32_t(const float *, int32_t, float)> callback_wrapper =
      [env, callback](const float *samples, int32_t n, float) -> int32_t {
    jfloatArray samples_arr = env->NewFloatArray(n);
    env->SetFloatArrayRegion(samples_arr, 0, n, samples);
    int32_t ret = CallCallback(env, callback, samples_arr);
    env->DeleteLocalRef(samples_arr);
    return ret;
  };

  auto audio = tts->Generate(p_text, gen_config, callback_wrapper);
  env->ReleaseStringUTFChars(text, p_text);

  return CreateAudioObject(env, audio.samples, audio.sample_rate);
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
