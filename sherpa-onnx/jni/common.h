// sherpa-onnx/jni/common.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_JNI_COMMON_H_
#define SHERPA_ONNX_JNI_COMMON_H_

#include <string>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if defined(_WIN32)
#if defined(SHERPA_ONNX_BUILD_SHARED_LIBS)
#define SHERPA_ONNX_EXPORT __declspec(dllexport)
#define SHERPA_ONNX_IMPORT __declspec(dllimport)
#else
#define SHERPA_ONNX_EXPORT
#define SHERPA_ONNX_IMPORT
#endif
#else  // WIN32
#define SHERPA_ONNX_EXPORT __attribute__((visibility("default")))

#define SHERPA_ONNX_IMPORT SHERPA_ONNX_EXPORT
#endif  // WIN32

#if defined(SHERPA_ONNX_BUILD_MAIN_LIB)
#define SHERPA_ONNX_API SHERPA_ONNX_EXPORT
#else
#define SHERPA_ONNX_API SHERPA_ONNX_IMPORT
#endif

// If you use ndk, you can find "jni.h" inside
// android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
#include "jni.h"  // NOLINT

#define SHERPA_ONNX_EXTERN_C extern "C" SHERPA_ONNX_API

#define SHERPA_ONNX_JNI_READ_STRING(cpp_field, kotlin_field, cls, config)     \
  do {                                                                        \
    jfieldID fid = env->GetFieldID(cls, #kotlin_field, "Ljava/lang/String;"); \
    if (fid == nullptr || env->ExceptionCheck()) {                            \
      SHERPA_ONNX_LOGE("Failed to get field ID for '%s'", #kotlin_field);     \
      SHERPA_ONNX_LOGE(                                                       \
          "Please check that your kotlin code matches the library file "      \
          "libsherpa-onnx-jni.so . If you are not sure, always use the "      \
          "LATEST code and the latest library");                              \
      if (env->ExceptionCheck()) {                                            \
        env->ExceptionDescribe();                                             \
        env->ExceptionClear();                                                \
      }                                                                       \
      jclass exClass = env->FindClass("java/lang/RuntimeException");          \
      if (exClass) {                                                          \
        env->ThrowNew(exClass, "Failed to get field ID for " #kotlin_field);  \
        env->DeleteLocalRef(exClass);                                         \
      }                                                                       \
      return ans;                                                             \
    }                                                                         \
    jstring s = (jstring)env->GetObjectField(config, fid);                    \
    if (s != nullptr) {                                                       \
      const char *p = env->GetStringUTFChars(s, nullptr);                     \
      cpp_field = p;                                                          \
      env->ReleaseStringUTFChars(s, p);                                       \
      env->DeleteLocalRef(s);                                                 \
    }                                                                         \
  } while (0)

#define SHERPA_ONNX_JNI_READ_FLOAT(cpp_field, kotlin_field, cls, config)     \
  do {                                                                       \
    jfieldID fid = env->GetFieldID(cls, #kotlin_field, "F");                 \
    if (fid == nullptr || env->ExceptionCheck()) {                           \
      SHERPA_ONNX_LOGE("Failed to get field ID for '%s'", #kotlin_field);    \
      SHERPA_ONNX_LOGE(                                                      \
          "Please check that your kotlin code matches the library file "     \
          "libsherpa-onnx-jni.so . If you are not sure, always use the "     \
          "LATEST code and the latest library");                             \
      if (env->ExceptionCheck()) {                                           \
        env->ExceptionDescribe();                                            \
        env->ExceptionClear();                                               \
      }                                                                      \
      jclass exClass = env->FindClass("java/lang/RuntimeException");         \
      if (exClass) {                                                         \
        env->ThrowNew(exClass, "Failed to get field ID for " #kotlin_field); \
        env->DeleteLocalRef(exClass);                                        \
      }                                                                      \
      return ans;                                                            \
    }                                                                        \
    cpp_field = env->GetFloatField(config, fid);                             \
  } while (0)

#define SHERPA_ONNX_JNI_READ_INT(cpp_field, kotlin_field, cls, config)       \
  do {                                                                       \
    jfieldID fid = env->GetFieldID(cls, #kotlin_field, "I");                 \
    if (fid == nullptr || env->ExceptionCheck()) {                           \
      SHERPA_ONNX_LOGE("Failed to get field ID for '%s'", #kotlin_field);    \
      SHERPA_ONNX_LOGE(                                                      \
          "Please check that your kotlin code matches the library file "     \
          "libsherpa-onnx-jni.so . If you are not sure, always use the "     \
          "LATEST code and the latest library");                             \
      if (env->ExceptionCheck()) {                                           \
        env->ExceptionDescribe();                                            \
        env->ExceptionClear();                                               \
      }                                                                      \
      jclass exClass = env->FindClass("java/lang/RuntimeException");         \
      if (exClass) {                                                         \
        env->ThrowNew(exClass, "Failed to get field ID for " #kotlin_field); \
        env->DeleteLocalRef(exClass);                                        \
      }                                                                      \
      return ans;                                                            \
    }                                                                        \
    cpp_field = env->GetIntField(config, fid);                               \
  } while (0)

#define SHERPA_ONNX_JNI_READ_BOOL(cpp_field, kotlin_field, cls, config)      \
  do {                                                                       \
    jfieldID fid = env->GetFieldID(cls, #kotlin_field, "Z");                 \
    if (fid == nullptr || env->ExceptionCheck()) {                           \
      SHERPA_ONNX_LOGE("Failed to get field ID for '%s'", #kotlin_field);    \
      SHERPA_ONNX_LOGE(                                                      \
          "Please check that your kotlin code matches the library file "     \
          "libsherpa-onnx-jni.so . If you are not sure, always use the "     \
          "LATEST code and the latest library");                             \
      if (env->ExceptionCheck()) {                                           \
        env->ExceptionDescribe();                                            \
        env->ExceptionClear();                                               \
      }                                                                      \
      jclass exClass = env->FindClass("java/lang/RuntimeException");         \
      if (exClass) {                                                         \
        env->ThrowNew(exClass, "Failed to get field ID for " #kotlin_field); \
        env->DeleteLocalRef(exClass);                                        \
      }                                                                      \
      return ans;                                                            \
    }                                                                        \
    cpp_field = env->GetBooleanField(config, fid);                           \
  } while (0)

// defined in jni.cc
jobject NewInteger(JNIEnv *env, int32_t value);
jobject NewFloat(JNIEnv *env, float value);

// Template function for non-void return types
template <typename Func, typename ReturnType>
ReturnType SafeJNI(JNIEnv *env, const char *functionName, Func func,
                   ReturnType defaultValue) {
  try {
    return func();
  } catch (const std::exception &e) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) + ": " + e.what();
      env->ThrowNew(exClass, errorMessage.c_str());
      env->DeleteLocalRef(exClass);
    }
  } catch (...) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) +
                                 ": Native exception: caught unknown exception";
      env->ThrowNew(exClass, errorMessage.c_str());
      env->DeleteLocalRef(exClass);
    }
  }
  return defaultValue;
}

// Specialization for void return type
template <typename Func>
void SafeJNI(JNIEnv *env, const char *functionName, Func func) {
  try {
    func();
  } catch (const std::exception &e) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) + ": " + e.what();
      env->ThrowNew(exClass, errorMessage.c_str());
      env->DeleteLocalRef(exClass);
    }
  } catch (...) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) +
                                 ": Native exception: caught unknown exception";
      env->ThrowNew(exClass, errorMessage.c_str());
      env->DeleteLocalRef(exClass);
    }
  }
}

// Helper function to validate JNI pointers
inline bool ValidatePointer(JNIEnv *env, jlong ptr, const char *functionName,
                            const char *message) {
  if (ptr == 0) {
    jclass exClass = env->FindClass("java/lang/NullPointerException");
    if (exClass != nullptr) {
      std::string errorMessage = std::string(functionName) + ": " + message;
      env->ThrowNew(exClass, errorMessage.c_str());
      env->DeleteLocalRef(exClass);
    }
    return false;
  }
  return true;
}

namespace sherpa_onnx {
void PrependAdspLibraryPath(const std::string &new_path);
}

#endif  // SHERPA_ONNX_JNI_COMMON_H_
