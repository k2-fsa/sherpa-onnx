// sherpa-onnx/csrc/macros.h
//
// Copyright      2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_MACROS_H_
#define SHERPA_ONNX_CSRC_MACROS_H_
#include <stdio.h>

#if __ANDROID_API__ >= 8
#include "android/log.h"
#define SHERPA_ONNX_LOGE(...)                                            \
  do {                                                                   \
    fprintf(stderr, "%s:%s:%d ", __FILE__, __func__,                     \
            static_cast<int>(__LINE__));                                 \
    fprintf(stderr, ##__VA_ARGS__);                                      \
    fprintf(stderr, "\n");                                               \
    __android_log_print(ANDROID_LOG_WARN, "sherpa-onnx", ##__VA_ARGS__); \
  } while (0)
#else
#define SHERPA_ONNX_LOGE(...)                        \
  do {                                               \
    fprintf(stderr, "%s:%s:%d ", __FILE__, __func__, \
            static_cast<int>(__LINE__));             \
    fprintf(stderr, ##__VA_ARGS__);                  \
    fprintf(stderr, "\n");                           \
  } while (0)
#endif

// Read an integer
#define SHERPA_ONNX_READ_META_DATA(dst, src_key)                        \
  do {                                                                  \
    auto value =                                                        \
        meta_data.LookupCustomMetadataMapAllocated(src_key, allocator); \
    if (!value) {                                                       \
      SHERPA_ONNX_LOGE("%s does not exist in the metadata", src_key);   \
      exit(-1);                                                         \
    }                                                                   \
                                                                        \
    dst = atoi(value.get());                                            \
    if (dst < 0) {                                                      \
      SHERPA_ONNX_LOGE("Invalid value %d for %s", dst, src_key);        \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

// read a vector of integers
#define SHERPA_ONNX_READ_META_DATA_VEC(dst, src_key)                     \
  do {                                                                   \
    auto value =                                                         \
        meta_data.LookupCustomMetadataMapAllocated(src_key, allocator);  \
    if (!value) {                                                        \
      SHERPA_ONNX_LOGE("%s does not exist in the metadata", src_key);    \
      exit(-1);                                                          \
    }                                                                    \
                                                                         \
    bool ret = SplitStringToIntegers(value.get(), ",", true, &dst);      \
    if (!ret) {                                                          \
      SHERPA_ONNX_LOGE("Invalid value %s for %s", value.get(), src_key); \
      exit(-1);                                                          \
    }                                                                    \
  } while (0)

// read a vector of floats
#define SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(dst, src_key)               \
  do {                                                                   \
    auto value =                                                         \
        meta_data.LookupCustomMetadataMapAllocated(src_key, allocator);  \
    if (!value) {                                                        \
      SHERPA_ONNX_LOGE("%s does not exist in the metadata", src_key);    \
      exit(-1);                                                          \
    }                                                                    \
                                                                         \
    bool ret = SplitStringToFloats(value.get(), ",", true, &dst);        \
    if (!ret) {                                                          \
      SHERPA_ONNX_LOGE("Invalid value %s for %s", value.get(), src_key); \
      exit(-1);                                                          \
    }                                                                    \
  } while (0)

// read a vector of strings
#define SHERPA_ONNX_READ_META_DATA_VEC_STRING(dst, src_key)                   \
  do {                                                                        \
    auto value =                                                              \
        meta_data.LookupCustomMetadataMapAllocated(src_key, allocator);       \
    if (!value) {                                                             \
      SHERPA_ONNX_LOGE("%s does not exist in the metadata", src_key);         \
      exit(-1);                                                               \
    }                                                                         \
    SplitStringToVector(value.get(), ",", false, &dst);                       \
                                                                              \
    if (dst.empty()) {                                                        \
      SHERPA_ONNX_LOGE("Invalid value %s for %s. Empty vector!", value.get(), \
                       src_key);                                              \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)

// Read a string
#define SHERPA_ONNX_READ_META_DATA_STR(dst, src_key)                    \
  do {                                                                  \
    auto value =                                                        \
        meta_data.LookupCustomMetadataMapAllocated(src_key, allocator); \
    if (!value) {                                                       \
      SHERPA_ONNX_LOGE("%s does not exist in the metadata", src_key);   \
      exit(-1);                                                         \
    }                                                                   \
                                                                        \
    dst = value.get();                                                  \
    if (dst.empty()) {                                                  \
      SHERPA_ONNX_LOGE("Invalid value for %s\n", src_key);              \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#endif  // SHERPA_ONNX_CSRC_MACROS_H_
