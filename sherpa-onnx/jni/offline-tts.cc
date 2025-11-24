// sherpa-onnx/jni/offline-tts.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/wave-writer.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

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


  // zipvoice
  fid = env->GetFieldID(model_config_cls, "zipvoice",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTtsZipvoiceModelConfig;");
  jobject zipvoice = env->GetObjectField(model, fid);
  jclass zipvoice_cls = env->GetObjectClass(zipvoice);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.zipvoice.tokens, tokens, zipvoice_cls, zipvoice);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.zipvoice.text_model, textModel, zipvoice_cls, zipvoice);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.zipvoice.flow_matching_model, flowMatchingModel, zipvoice_cls, zipvoice);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.zipvoice.vocoder, vocoder, zipvoice_cls, zipvoice);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.zipvoice.data_dir, dataDir, zipvoice_cls, zipvoice);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.zipvoice.pinyin_dict, pinyinDict, zipvoice_cls, zipvoice);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.zipvoice.feat_scale, featScale, zipvoice_cls, zipvoice);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.zipvoice.t_shift, tShift, zipvoice_cls, zipvoice);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.zipvoice.target_rms, targetRms, zipvoice_cls, zipvoice);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.model.zipvoice.guidance_scale, guidanceScale, zipvoice_cls, zipvoice);


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
Java_com_k2fsa_sherpa_onnx_OfflineTts_generateWithPromptImpl(JNIEnv *env, jobject /*obj*/,
                                                   jlong ptr, jstring text, jstring prompt_text, jfloatArray prompt_samples, jint sample_rate, jfloat speed, jint num_step) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);
  const char *p_prompt_text = env->GetStringUTFChars(prompt_text, nullptr);

  jfloat *prompt_samples_elements = env->GetFloatArrayElements(prompt_samples, nullptr);
  jsize prompt_samples_n = env->GetArrayLength(prompt_samples);
  std::vector<float> prompt_samples_vec(prompt_samples_elements, prompt_samples_elements + prompt_samples_n);

  auto audio = reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr)->Generate(p_text, p_prompt_text, prompt_samples_vec, sample_rate, speed, num_step);

  jfloatArray samples_arr = env->NewFloatArray(audio.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, audio.samples.size(),
                           audio.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, audio.sample_rate));

  env->ReleaseStringUTFChars(text, p_text);
  env->ReleaseStringUTFChars(prompt_text, p_prompt_text);
  env->ReleaseFloatArrayElements(prompt_samples, prompt_samples_elements, JNI_ABORT);

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
    jclass cls = env->GetObjectClass(callback);

#if 0
        // this block is for debugging only
        // see also
        // https://jnjosh.com/posts/kotlinfromcpp/
        jmethodID classMethodId =
            env->GetMethodID(cls, "getClass", "()Ljava/lang/Class;");
        jobject klassObj = env->CallObjectMethod(callback, classMethodId);
        auto klassObject = env->GetObjectClass(klassObj);
        auto nameMethodId =
            env->GetMethodID(klassObject, "getName", "()Ljava/lang/String;");
        jstring classString =
            (jstring)env->CallObjectMethod(klassObj, nameMethodId);
        auto className = env->GetStringUTFChars(classString, NULL);
        SHERPA_ONNX_LOGE("name is: %s", className);
        env->ReleaseStringUTFChars(classString, className);
#endif

    jmethodID mid = env->GetMethodID(cls, "invoke", "([F)Ljava/lang/Integer;");
    if (mid == nullptr) {
      SHERPA_ONNX_LOGE("Failed to get the callback. Ignore it.");
      return 1;
    }

    jfloatArray samples_arr = env->NewFloatArray(n);
    env->SetFloatArrayRegion(samples_arr, 0, n, samples);

    jobject should_continue = env->CallObjectMethod(callback, mid, samples_arr);
    jclass jklass = env->GetObjectClass(should_continue);
    jmethodID int_value_mid = env->GetMethodID(jklass, "intValue", "()I");
    return env->CallIntMethod(should_continue, int_value_mid);
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
Java_com_k2fsa_sherpa_onnx_OfflineTts_generateWithPromptWithCallbackImpl(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring text, jstring prompt_text, jfloatArray prompt_samples, jint sample_rate,
    jfloat speed, jint num_step, jobject callback) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);
  const char *p_prompt_text = env->GetStringUTFChars(prompt_text, nullptr);

  jfloat *prompt_samples_elements = env->GetFloatArrayElements(prompt_samples, nullptr);
  jsize prompt_samples_n = env->GetArrayLength(prompt_samples);
  std::vector<float> prompt_samples_vec(prompt_samples_elements, prompt_samples_elements + prompt_samples_n);

  std::function<int32_t(const float *, int32_t, float)> callback_wrapper =
      [env, callback](const float *samples, int32_t n,
                      float /*progress*/) -> int {
    jclass cls = env->GetObjectClass(callback);

#if 0
        // this block is for debugging only
        // see also
        // https://jnjosh.com/posts/kotlinfromcpp/
        jmethodID classMethodId =
            env->GetMethodID(cls, "getClass", "()Ljava/lang/Class;");
        jobject klassObj = env->CallObjectMethod(callback, classMethodId);
        auto klassObject = env->GetObjectClass(klassObj);
        auto nameMethodId =
            env->GetMethodID(klassObject, "getName", "()Ljava/lang/String;");
        jstring classString =
            (jstring)env->CallObjectMethod(klassObj, nameMethodId);
        auto className = env->GetStringUTFChars(classString, NULL);
        SHERPA_ONNX_LOGE("name is: %s", className);
        env->ReleaseStringUTFChars(classString, className);
#endif

    jmethodID mid = env->GetMethodID(cls, "invoke", "([F)Ljava/lang/Integer;");
    if (mid == nullptr) {
      SHERPA_ONNX_LOGE("Failed to get the callback. Ignore it.");
      return 1;
    }

    jfloatArray samples_arr = env->NewFloatArray(n);
    env->SetFloatArrayRegion(samples_arr, 0, n, samples);

    jobject should_continue = env->CallObjectMethod(callback, mid, samples_arr);
    jclass jklass = env->GetObjectClass(should_continue);
    jmethodID int_value_mid = env->GetMethodID(jklass, "intValue", "()I");
    return env->CallIntMethod(should_continue, int_value_mid);
  };

  auto tts = reinterpret_cast<sherpa_onnx::OfflineTts *>(ptr);
  auto audio = tts->Generate(p_text, p_prompt_text, prompt_samples_vec, sample_rate, speed, num_step, callback_wrapper);

  jfloatArray samples_arr = env->NewFloatArray(audio.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, audio.samples.size(),
                           audio.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, audio.sample_rate));

  env->ReleaseStringUTFChars(text, p_text);
  env->ReleaseStringUTFChars(prompt_text, p_prompt_text);
  env->ReleaseFloatArrayElements(prompt_samples, prompt_samples_elements, JNI_ABORT);

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
