// sherpa-onnx/csrc/macros.h
//
// Copyright      2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_MACROS_H_
#define SHERPA_ONNX_CSRC_RKNN_MACROS_H_

#include "sherpa-onnx/csrc/macros.h"

#define SHERPA_ONNX_RKNN_CHECK(ret, msg, ...)      \
  do {                                             \
    if (ret != RKNN_SUCC) {                        \
      SHERPA_ONNX_LOGE("Return code is: %d", ret); \
      SHERPA_ONNX_LOGE(msg, ##__VA_ARGS__);        \
      SHERPA_ONNX_EXIT(-1);                        \
    }                                              \
  } while (0)

#endif  // SHERPA_ONNX_CSRC_RKNN_MACROS_H_
