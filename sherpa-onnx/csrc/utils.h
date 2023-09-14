// sherpa-onnx/csrc/utils.h
//
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_UTILS_H_
#define SHERPA_ONNX_CSRC_UTILS_H_

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

/* Encode the hotwords in an input stream to be tokens ids.
 *
 * @param is The input stream, it contains several lines, one hotword for each
 *           line. For each hotword, the tokens (cjkchar or bpe) are separated
 *           by spaces.
 * @param symbol_table  The tokens table mapping symbols to ids. All the symbols
 *                      in the stream should be in the symbol_table, if not this
 *                      function returns fasle.
 *
 * @@param hotwords  The encoded ids to be written to.
 *
 * @return  If all the symbols from ``is`` are in the symbol_table, returns true
 *          otherwise returns false.
 */
bool EncodeHotwords(std::istream &is, const SymbolTable &symbol_table,
                    std::vector<std::vector<int32_t>> *hotwords);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_UTILS_H_
