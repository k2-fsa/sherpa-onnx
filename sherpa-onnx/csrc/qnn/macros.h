// sherpa-onnx/csrc/qnn/macros.h
//
// Copyright      2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_MACROS_H_
#define SHERPA_ONNX_CSRC_QNN_MACROS_H_

#include "sherpa-onnx/csrc/macros.h"

#define SHERPA_ONNX_QNN_CHECK(ret, msg, ...)                             \
  do {                                                                   \
    if (ret != QNN_SUCCESS) {                                            \
      SHERPA_ONNX_LOGE("Return code is: %d", static_cast<int32_t>(ret)); \
      SHERPA_ONNX_LOGE(msg, ##__VA_ARGS__);                              \
      SHERPA_ONNX_EXIT(-1);                                              \
    }                                                                    \
  } while (0)

#endif  // SHERPA_ONNX_CSRC_QNN_MACROS_H_
