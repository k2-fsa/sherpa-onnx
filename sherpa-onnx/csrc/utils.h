// sherpa-onnx/csrc/utils.h
//
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_UTILS_H_
#define SHERPA_ONNX_CSRC_UTILS_H_

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

bool EncodeHotwords(std::istream &is, const SymbolTable &symbol_table,
                    std::vector<std::vector<int32_t>> *hotwords);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_UTILS_H_
