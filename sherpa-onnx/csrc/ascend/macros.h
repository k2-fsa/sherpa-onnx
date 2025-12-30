// sherpa-onnx/csrc/ascend/macros.h
//
// Copyright      2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ASCEND_MACROS_H_
#define SHERPA_ONNX_CSRC_ASCEND_MACROS_H_

#include "sherpa-onnx/csrc/macros.h"

#define SHERPA_ONNX_ASCEND_CHECK(ret, msg, ...)    \
  do {                                             \
    if (ret != ACL_ERROR_NONE) {                   \
      const char *_msg = aclGetRecentErrMsg();     \
      SHERPA_ONNX_LOGE("Return code is: %d", ret); \
      SHERPA_ONNX_LOGE("Error message: %s", _msg); \
      SHERPA_ONNX_LOGE(msg, ##__VA_ARGS__);        \
      SHERPA_ONNX_EXIT(-1);                        \
    }                                              \
  } while (0)

#endif  // SHERPA_ONNX_CSRC_ASCEND_MACROS_H_
