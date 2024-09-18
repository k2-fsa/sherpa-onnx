// scripts/node-addon-api/src/macros.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SCRIPTS_NODE_ADDON_API_SRC_MACROS_H_
#define SCRIPTS_NODE_ADDON_API_SRC_MACROS_H_

#include <algorithm>
#include <string>

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

#endif  // SCRIPTS_NODE_ADDON_API_SRC_MACROS_H_
