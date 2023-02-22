// sherpa-onnx/csrc/symbol-table.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/symbol-table.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <strstream>

#include "sherpa-onnx/csrc/onnx-utils.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

namespace sherpa_onnx {

SymbolTable::SymbolTable(const std::string &filename) {
  std::ifstream is(filename);
  Init(is);
}

#if __ANDROID_API__ >= 9
SymbolTable::SymbolTable(AAssetManager *mgr, const std::string &filename) {
  auto buf = ReadFile(mgr, filename);

  std::istrstream is(buf.data(), buf.size());
  Init(is);
}
#endif

void SymbolTable::Init(std::istream &is) {
  std::string sym;
  int32_t id;
  while (is >> sym >> id) {
    if (sym.size() >= 3) {
      // For BPE-based models, we replace ▁ with a space
      // Unicode 9601, hex 0x2581, utf8 0xe29681
      const uint8_t *p = reinterpret_cast<const uint8_t *>(sym.c_str());
      if (p[0] == 0xe2 && p[1] == 0x96 && p[2] == 0x81) {
        sym = sym.replace(0, 3, " ");
      }
    }

    assert(!sym.empty());
    assert(sym2id_.count(sym) == 0);
    assert(id2sym_.count(id) == 0);

    sym2id_.insert({sym, id});
    id2sym_.insert({id, sym});
  }
  assert(is.eof());
}

std::string SymbolTable::ToString() const {
  std::ostringstream os;
  char sep = ' ';
  for (const auto &p : sym2id_) {
    os << p.first << sep << p.second << "\n";
  }
  return os.str();
}

const std::string &SymbolTable::operator[](int32_t id) const {
  return id2sym_.at(id);
}

int32_t SymbolTable::operator[](const std::string &sym) const {
  return sym2id_.at(sym);
}

bool SymbolTable::contains(int32_t id) const { return id2sym_.count(id) != 0; }

bool SymbolTable::contains(const std::string &sym) const {
  return sym2id_.count(sym) != 0;
}

std::ostream &operator<<(std::ostream &os, const SymbolTable &symbol_table) {
  return os << symbol_table.ToString();
}

}  // namespace sherpa_onnx
