// sherpa-onnx/csrc/jieba.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_JIEBA_H_
#define SHERPA_ONNX_CSRC_JIEBA_H_

#include <memory>
#include <string>

#include "cppjieba/Jieba.hpp"

namespace sherpa_onnx {

std::unique_ptr<cppjieba::Jieba> InitJieba(const std::string &dict_dir);
}

#endif  // SHERPA_ONNX_CSRC_JIEBA_H_
