// sherpa-onnx/csrc/symbol-table.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

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
