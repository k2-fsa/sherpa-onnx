// scripts/node-addon-api/src/macros.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SCRIPTS_NODE_ADDON_API_SRC_MACROS_H_
#define SCRIPTS_NODE_ADDON_API_SRC_MACROS_H_

#include <algorithm>
#include <string>

#if __OHOS__
#include "hilog/log.h"
#include "rawfile/raw_file_manager.h"

#undef LOG_DOMAIN
#undef LOG_TAG

// https://gitee.com/openharmony/docs/blob/145a084f0b742e4325915e32f8184817927d1251/en/contribute/OpenHarmony-Log-guide.md#hilog-api-usage-specifications
#define LOG_DOMAIN 0x6666
#define LOG_TAG "sherpa_onnx"
#endif

#define SHERPA_ONNX_ASSIGN_ATTR_STR(c_name, js_name)                       \
  do {                                                                     \
    if (o.Has(#js_name) && o.Get(#js_name).IsString()) {                   \
      Napi::String _str = o.Get(#js_name).As<Napi::String>();              \
      std::string s = _str.Utf8Value();                                    \
      char *p = new char[s.size() + 1];                                    \
      std::copy(s.begin(), s.end(), p);                                    \
      p[s.size()] = 0;                                                     \
                                                                           \
      c.c_name = p;                                                        \
    } else if (o.Has(#js_name) && o.Get(#js_name).IsTypedArray()) {        \
      Napi::Uint8Array _array = o.Get(#js_name).As<Napi::Uint8Array>();    \
      char *p = new char[_array.ElementLength() + 1];                      \
      std::copy(_array.Data(), _array.Data() + _array.ElementLength(), p); \
      p[_array.ElementLength()] = '\0';                                    \
                                                                           \
      c.c_name = p;                                                        \
    }                                                                      \
  } while (0)

#define SHERPA_ONNX_ASSIGN_ATTR_INT32(c_name, js_name)            \
  do {                                                            \
    if (o.Has(#js_name) && o.Get(#js_name).IsNumber()) {          \
      c.c_name = o.Get(#js_name).As<Napi::Number>().Int32Value(); \
    }                                                             \
  } while (0)

#define SHERPA_ONNX_ASSIGN_ATTR_FLOAT(c_name, js_name)            \
  do {                                                            \
    if (o.Has(#js_name) && o.Get(#js_name).IsNumber()) {          \
      c.c_name = o.Get(#js_name).As<Napi::Number>().FloatValue(); \
    }                                                             \
  } while (0)

#define SHERPA_ONNX_DELETE_C_STR(p) \
  do {                              \
    if (p) {                        \
      delete[] p;                   \
    }                               \
  } while (0)

#endif  // SCRIPTS_NODE_ADDON_API_SRC_MACROS_H_
