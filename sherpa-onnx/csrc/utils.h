// sherpa-onnx/csrc/utils.h
//
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_UTILS_H_
#define SHERPA_ONNX_CSRC_UTILS_H_

#include <string>
#include <vector>

#include "sentencepiece_processor.h"  //NOLINT
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

bool EncodeWithBpe(const std::string word,
                   const sentencepiece::SentencePieceProcessor &bpe_processor,
                   std::vector<std::string> *syms);

bool EncodeHotwords(std::istream &is, const std::string &tokens_type,
                    const SymbolTable &symbol_table,
                    const sentencepiece::SentencePieceProcessor &bpe_processor,
                    std::vector<std::vector<int32_t>> *hotwords);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_UTILS_H_
