// scripts/node-addon-api/src/macros.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SCRIPTS_NODE_ADDON_API_SRC_MACROS_H_
#define SCRIPTS_NODE_ADDON_API_SRC_MACROS_H_

#if __OHOS__
#include "hilog/log.h"
#include "rawfile/raw_file_manager.h"

#undef LOG_DOMAIN
#undef LOG_TAG

// https://gitee.com/openharmony/docs/blob/145a084f0b742e4325915e32f8184817927d1251/en/contribute/OpenHarmony-Log-guide.md#hilog-api-usage-specifications
#define LOG_DOMAIN 0x6666
#define LOG_TAG "sherpa_onnx"
#endif

#define SHERPA_ONNX_ASSIGN_ATTR_STR(c_name, js_name)                                                                   \
    do {                                                                                                               \
        if (o.Has(#js_name)) {                                                                                         \
            Napi::Value _value = o.Get(#js_name);                                                                      \
            if (_value.IsString()) {                                                                                   \
                Napi::String _str = _value.As<Napi::String>();                                                         \
                std::string s = _str.Utf8Value();                                                                      \
                char *p = new char[s.size() + 1];                                                                      \
                std::copy(s.begin(), s.end(), p);                                                                      \
                p[s.size()] = 0;                                                                                       \
                                                                                                                       \
                c.c_name = p;                                                                                          \
            } else if (_value.IsTypedArray()) {                                                                        \
                Napi::Uint8Array _array = _value.As<Napi::Uint8Array>();                                               \
                char *p = new char[_array.ElementLength() + 1];                                                        \
                std::copy(_array.Data(), _array.Data() + _array.ElementLength(), p);                                   \
                p[_array.ElementLength()] = '\0';                                                                      \
                                                                                                                       \
                c.c_name = p;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#define SHERPA_ONNX_ASSIGN_ATTR_INT32(c_name, js_name)                                                                 \
    do {                                                                                                               \
        if (o.Has(#js_name)) {                                                                                         \
            Napi::Value _value = o.Get(#js_name);                                                                      \
            if (_value.IsNumber()) {                                                                                   \
                c.c_name = _value.As<Napi::Number>().Int32Value();                                                     \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#define SHERPA_ONNX_ASSIGN_ATTR_FLOAT(c_name, js_name)                                                                 \
    do {                                                                                                               \
        if (o.Has(#js_name)) {                                                                                         \
            Napi::Value _value = o.Get(#js_name);                                                                      \
            if (_value.IsNumber()) {                                                                                   \
                c.c_name = _value.As<Napi::Number>().FloatValue();                                                     \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

// js_name may be either a boolean or a number; a number is taken as an int32
#define SHERPA_ONNX_ASSIGN_ATTR_BOOL(c_name, js_name)                                                                  \
    do {                                                                                                               \
        if (o.Has(#js_name)) {                                                                                         \
            Napi::Value _value = o.Get(#js_name);                                                                      \
            if (_value.IsBoolean()) {                                                                                  \
                c.c_name = _value.As<Napi::Boolean>().Value();                                                         \
            } else if (_value.IsNumber()) {                                                                            \
                c.c_name = _value.As<Napi::Number>().Int32Value();                                                     \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#define SHERPA_ONNX_DELETE_C_STR(p)                                                                                    \
    do {                                                                                                               \
        if (p) {                                                                                                       \
            delete[] p;                                                                                                \
        }                                                                                                              \
    } while (0)

#endif // SCRIPTS_NODE_ADDON_API_SRC_MACROS_H_
