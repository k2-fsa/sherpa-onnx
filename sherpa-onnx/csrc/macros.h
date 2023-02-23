
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
    fprintf(stderr, ##__VA_ARGS__);                                      \
    fprintf(stderr, "\n");                                               \
    __android_log_print(ANDROID_LOG_WARN, "sherpa-onnx", ##__VA_ARGS__); \
  } while (0)
#else
#define SHERPA_ONNX_LOGE(...)       \
  do {                              \
    fprintf(stderr, ##__VA_ARGS__); \
    fprintf(stderr, "\n");          \
  } while (0)
#endif

#define SHERPA_ONNX_READ_META_DATA(dst, src_key)                        \
  do {                                                                  \
    auto value =                                                        \
        meta_data.LookupCustomMetadataMapAllocated(src_key, allocator); \
    if (!value) {                                                       \
      fprintf(stderr, "%s does not exist in the metadata\n", src_key);  \
      exit(-1);                                                         \
    }                                                                   \
                                                                        \
    dst = atoi(value.get());                                            \
    if (dst <= 0) {                                                     \
      fprintf(stderr, "Invalid value %d for %s\n", dst, src_key);       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define SHERPA_ONNX_READ_META_DATA_VEC(dst, src_key)                      \
  do {                                                                    \
    auto value =                                                          \
        meta_data.LookupCustomMetadataMapAllocated(src_key, allocator);   \
    if (!value) {                                                         \
      fprintf(stderr, "%s does not exist in the metadata\n", src_key);    \
      exit(-1);                                                           \
    }                                                                     \
                                                                          \
    bool ret = SplitStringToIntegers(value.get(), ",", true, &dst);       \
    if (!ret) {                                                           \
      fprintf(stderr, "Invalid value %s for %s\n", value.get(), src_key); \
      exit(-1);                                                           \
    }                                                                     \
  } while (0)

#endif  // SHERPA_ONNX_CSRC_MACROS_H_
