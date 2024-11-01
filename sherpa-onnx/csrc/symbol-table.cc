// sherpa-onnx/csrc/symbol-table.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/symbol-table.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/base64-decode.h"
#include "sherpa-onnx/csrc/lexicon.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

namespace {
// copied from
// https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring
const char *ws = " \t\n\r\f\v";

// trim from end of string (right)
inline void TrimRight(std::string *s, const char *t = ws) {
  s->erase(s->find_last_not_of(t) + 1);
}

// trim from beginning of string (left)
inline void TrimLeft(std::string *s, const char *t = ws) {
  s->erase(0, s->find_first_not_of(t));
}

// trim from both ends of string (right then left)
inline void Trim(std::string *s, const char *t = ws) {
  TrimRight(s, t);
  TrimLeft(s, t);
}
}  // namespace

std::unordered_map<std::string, int32_t> ReadTokens(
    std::istream &is,
    std::unordered_map<int32_t, std::string> *id2token /*= nullptr*/) {
  std::unordered_map<std::string, int32_t> token2id;

  std::string line;

  std::string sym;
  int32_t id = -1;
  while (std::getline(is, line)) {
    Trim(&line);
    std::istringstream iss(line);
    iss >> sym;
    if (iss.eof()) {
      id = atoi(sym.c_str());
      sym = " ";
    } else {
      iss >> id;
    }

    // eat the trailing \r\n on windows
    iss >> std::ws;
    if (!iss.eof()) {
      SHERPA_ONNX_LOGE("Error: %s", line.c_str());
      exit(-1);
    }

#if 0
    if (token2id.count(sym)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(sym));
      exit(-1);
    }
#endif
    if (id2token) {
      id2token->insert({id, sym});
    }

    token2id.insert({std::move(sym), id});
  }

  return token2id;
}

SymbolTable::SymbolTable(const std::string &filename, bool is_file) {
  if (is_file) {
    std::ifstream is(filename);
    Init(is);
  } else {
    std::istringstream iss(filename);
    Init(iss);
  }
}

#if __ANDROID_API__ >= 9
SymbolTable::SymbolTable(AAssetManager *mgr, const std::string &filename) {
  auto buf = ReadFile(mgr, filename);

  std::istrstream is(buf.data(), buf.size());
  Init(is);
}
#endif

void SymbolTable::Init(std::istream &is) { sym2id_ = ReadTokens(is, &id2sym_); }

std::string SymbolTable::ToString() const {
  std::ostringstream os;
  char sep = ' ';
  for (const auto &p : sym2id_) {
    os << p.first << sep << p.second << "\n";
  }
  return os.str();
}

const std::string SymbolTable::operator[](int32_t id) const {
  std::string sym = id2sym_.at(id);
  if (sym.size() >= 3) {
    // For BPE-based models, we replace ▁ with a space
    // Unicode 9601, hex 0x2581, utf8 0xe29681
    const uint8_t *p = reinterpret_cast<const uint8_t *>(sym.c_str());
    if (p[0] == 0xe2 && p[1] == 0x96 && p[2] == 0x81) {
      sym = sym.replace(0, 3, " ");
    }
  }

  // for byte-level BPE
  // id 0 is blank, id 1 is sos/eos, id 2 is unk
  //
  // Note: For moonshine models, 0 is <unk>, 1, is <s>, 2 is</s>
  if (id >= 3 && id <= 258 && sym.size() == 6 && sym[0] == '<' &&
      sym[1] == '0' && sym[2] == 'x' && sym[5] == '>') {
    std::ostringstream os;
    os << std::hex << std::uppercase << (id - 3);

    if (std::string(sym.data() + 3, sym.data() + 5) == os.str()) {
      uint8_t i = id - 3;
      sym = std::string(&i, &i + 1);
    }
  }
  return sym;
}

int32_t SymbolTable::operator[](const std::string &sym) const {
  return sym2id_.at(sym);
}

bool SymbolTable::Contains(int32_t id) const { return id2sym_.count(id) != 0; }

bool SymbolTable::Contains(const std::string &sym) const {
  return sym2id_.count(sym) != 0;
}

std::ostream &operator<<(std::ostream &os, const SymbolTable &symbol_table) {
  return os << symbol_table.ToString();
}

void SymbolTable::ApplyBase64Decode() {
  sym2id_.clear();
  for (auto &p : id2sym_) {
    p.second = Base64Decode(p.second);
    sym2id_[p.second] = p.first;
  }
}

}  // namespace sherpa_onnx
