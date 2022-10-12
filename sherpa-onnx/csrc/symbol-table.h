/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SHERPA_ONNX_CSRC_SYMBOL_TABLE_H_
#define SHERPA_ONNX_CSRC_SYMBOL_TABLE_H_

#include <string>
#include <unordered_map>

namespace sherpa_onnx {

/// It manages mapping between symbols and integer IDs.
class SymbolTable {
 public:
  SymbolTable() = default;
  /// Construct a symbol table from a file.
  /// Each line in the file contains two fields:
  ///
  ///    sym ID
  ///
  /// Fields are separated by space(s).
  explicit SymbolTable(const std::string &filename);

  /// Return a string representation of this symbol table
  std::string ToString() const;

  /// Return the symbol corresponding to the given ID.
  const std::string &operator[](int32_t id) const;
  /// Return the ID corresponding to the given symbol.
  int32_t operator[](const std::string &sym) const;

  /// Return true if there is a symbol with the given ID.
  bool contains(int32_t id) const;

  /// Return true if there is a given symbol in the symbol table.
  bool contains(const std::string &sym) const;

 private:
  std::unordered_map<std::string, int32_t> sym2id_;
  std::unordered_map<int32_t, std::string> id2sym_;
};

std::ostream &operator<<(std::ostream &os, const SymbolTable &symbol_table);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SYMBOL_TABLE_H_
