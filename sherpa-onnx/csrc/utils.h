// sherpa-onnx/csrc/utils.h
//
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_UTILS_H_
#define SHERPA_ONNX_CSRC_UTILS_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "sherpa-onnx/csrc/symbol-table.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_onnx {

/* Return a copy of 'in' with duplicate elements removed.
 *
 * First occurrence of each element is kept.
 *
 * @param in The input vector to deduplicate. Must be hashable.
 * @return A new vector containing the unique elements of 'in' in their
 *         original order of first appearance.
 */
template <typename T>
std::vector<T> Deduplicate(const std::vector<T> &in) {
  std::vector<T> out;
  out.reserve(in.size());
  std::unordered_set<T> seen;
  seen.reserve(in.size());
  for (const auto &x : in) {
    if (seen.insert(x).second) out.push_back(x);
  }
  return out;
}

/* Encode the hotwords in an input stream to be tokens ids.
 *
 * @param is The input stream, it contains several lines, one hotword for each
 *           line. For each hotword, the tokens (cjkchar or bpe) are separated
 *           by spaces.
 * @param symbol_table  The tokens table mapping symbols to ids. All the symbols
 *                      in the stream should be in the symbol_table, if not this
 *                      function returns false.
 *
 * @@param hotwords  The encoded ids to be written to.
 *
 * @return  If all the symbols from ``is`` are in the symbol_table, returns true
 *          otherwise returns false.
 */
bool EncodeHotwords(std::istream &is, const std::string &modeling_unit,
                    const SymbolTable &symbol_table,
                    const ssentencepiece::Ssentencepiece *bpe_encoder,
                    std::vector<std::vector<int32_t>> *hotwords_id,
                    std::vector<float> *boost_scores);

/* Encode the keywords in an input stream to be tokens ids.
 *
 * @param is The input stream, it contains several lines, one hotword for each
 *           line. For each hotword, the tokens (cjkchar or bpe) are separated
 *           by spaces, it might contain boosting score (starting with :),
 *           triggering threshold (starting with #) and keyword string (starting
 *           with @) too.
 * @param symbol_table  The tokens table mapping symbols to ids. All the symbols
 *                      in the stream should be in the symbol_table, if not this
 *                      function returns false.
 *
 * @param keywords_id The encoded ids to be written to.
 * @param keywords The original keyword string to be written to.
 * @param boost_scores  The boosting score for each keyword to be written to.
 * @param threshold  The triggering threshold for each keyword to be written to.
 *
 * @return  If all the symbols from ``is`` are in the symbol_table, returns true
 *          otherwise returns false.
 */
bool EncodeKeywords(std::istream &is, const SymbolTable &symbol_table,
                    std::vector<std::vector<int32_t>> *keywords_id,
                    std::vector<std::string> *keywords,
                    std::vector<float> *boost_scores,
                    std::vector<float> *threshold);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_UTILS_H_
