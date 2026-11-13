// sherpa-onnx/csrc/fa-lexicon.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/fa-lexicon.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

bool FaLexicon::Contains(const std::string &key) const {
  return entries.count(key) != 0;
}

const std::vector<FaPronunciation> &FaLexicon::Get(
    const std::string &key) const {
  static const std::vector<FaPronunciation> kEmpty;
  auto it = entries.find(key);
  if (it == entries.end()) {
    return kEmpty;
  }
  return it->second;
}

static bool IsNumericToken(const std::string &s) {
  float value;
  return ConvertStringToReal(s, &value);
}

FaLexicon FaLexicon::Load(const std::string &filename) {
  std::ifstream is = OpenInputFile(filename);
  if (!is) {
    throw std::runtime_error("Failed to open lexicon: '" + filename + "'");
  }

  FaLexicon lexicon;
  std::string line;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) {
      // Skip optional probability columns between the key and the phones.
      if (IsNumericToken(tok)) {
        continue;
      }
      tokens.push_back(tok);
    }
    if (tokens.size() < 2) {
      continue;
    }

    // "key [key2 ...] || phones || phones || ...": everything before the
    // first "||" is the key (it may contain spaces), and the rest is split
    // into one phone group per character/word of the key. A line without
    // "||" is "key phones" with a single group (external dictionaries).
    size_t bar = tokens.size();
    for (size_t i = 0; i != tokens.size(); ++i) {
      if (tokens[i] == "||") {
        bar = i;
        break;
      }
    }

    std::string key;
    FaPronunciation pron;
    if (bar == tokens.size()) {
      key = tokens[0];
      pron.groups.emplace_back(tokens.begin() + 1, tokens.end());
    } else {
      if (bar == 0) {
        continue;
      }
      for (size_t i = 0; i != bar; ++i) {
        if (i) {
          key += " ";
        }
        key += tokens[i];
      }
      pron.groups.emplace_back();
      for (size_t i = bar + 1; i != tokens.size(); ++i) {
        if (tokens[i] == "||") {
          pron.groups.emplace_back();
          continue;
        }
        pron.groups.back().push_back(tokens[i]);
      }
    }
    // Drop empty groups (e.g. a trailing "||").
    for (size_t i = pron.groups.size(); i > 0; --i) {
      if (pron.groups[i - 1].empty() && pron.groups.size() > 1) {
        pron.groups.erase(pron.groups.begin() + (i - 1));
      }
    }
    if (pron.groups.empty() || pron.groups[0].empty()) {
      continue;
    }
    for (const auto &g : pron.groups) {
      pron.phones.insert(pron.phones.end(), g.begin(), g.end());
    }

    auto &bucket = lexicon.entries[key];
    bool duplicate = false;
    for (auto &e : bucket) {
      if (e.phones == pron.phones) {
        duplicate = true;
        // Same pronunciation: keep the form with more per-character groups
        // so the key stays usable as a pronunciation context.
        if (pron.groups.size() > e.groups.size()) {
          e = std::move(pron);
        }
        break;
      }
    }
    if (duplicate) {
      SHERPA_ONNX_LOGE("duplicate pronunciation for '%s' in '%s'; ignored",
                       key.c_str(), filename.c_str());
    } else {
      bucket.push_back(std::move(pron));
    }
  }

  if (lexicon.entries.empty()) {
    throw std::runtime_error("Lexicon contains no valid entry: '" + filename +
                             "'");
  }
  return lexicon;
}

static bool IsSpaceCodepoint(char32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x3000;
}

// Everything that cannot appear inside a word (letters, marks, numbers) is
// treated as punctuation: dropped by the per-character tokenizer and
// stripped from word edges by the whitespace tokenizer.
static bool IsFaPunctuation(char32_t cp) {
  return !IsLexiconWordChar(cp) && !IsSpaceCodepoint(cp);
}

static bool HasHanziKey(const FaLexicon &lexicon) {
  for (const auto &kv : lexicon.entries) {
    if (ContainsCJK(kv.first)) {
      return true;
    }
  }
  return false;
}

// One token per character; whitespace and punctuation are dropped.
static std::vector<std::string> TokenizePerCharacter(
    const std::u32string &text) {
  std::vector<std::string> words;
  for (char32_t cp : text) {
    if (IsSpaceCodepoint(cp) || IsFaPunctuation(cp)) {
      continue;
    }
    words.push_back(Utf32ToUtf8(cp));
  }
  return words;
}

// Normalizes a raw whitespace-separated word: punctuation is stripped from
// both edges, letters are lowercased, and empty words (pure punctuation) are
// dropped.
static void AppendWord(const std::u32string &raw,
                       std::vector<std::string> *words) {
  size_t b = 0;
  size_t e = raw.size();
  while (b < e && IsFaPunctuation(raw[b])) {
    ++b;
  }
  while (e > b && IsFaPunctuation(raw[e - 1])) {
    --e;
  }
  std::string out;
  for (size_t i = b; i != e; ++i) {
    out += Utf32ToUtf8(ToSimpleLower(raw[i]));
  }
  if (!out.empty()) {
    words->push_back(std::move(out));
  }
}

// Splits on whitespace; see AppendWord() for word normalization.
static std::vector<std::string> TokenizeOnWhitespace(
    const std::u32string &text) {
  std::vector<std::string> words;
  std::u32string raw;
  for (char32_t cp : text) {
    if (IsSpaceCodepoint(cp)) {
      AppendWord(raw, &words);
      raw.clear();
    } else {
      raw += cp;
    }
  }
  AppendWord(raw, &words);
  return words;
}

std::vector<std::string> TokenizeFaText(const std::string &text,
                                        const FaLexicon &lexicon) {
  std::u32string cps = Utf8ToUtf32(text);
  bool hanzi = ContainsCJK(cps) && HasHanziKey(lexicon);
  return hanzi ? TokenizePerCharacter(cps) : TokenizeOnWhitespace(cps);
}

std::vector<std::vector<std::vector<std::string>>> ResolveFaPronunciations(
    const std::vector<std::string> &words, const FaLexicon &lexicon) {
  size_t n = words.size();
  size_t max_key_len = 1;
  for (const auto &kv : lexicon.entries) {
    size_t len = Utf8ToUtf32(kv.first).size();
    max_key_len = std::max(max_key_len, len);
  }

  // Per-token readings given by the " || " groups of the multi-token key
  // contexts covering the token (union over contexts).
  std::vector<std::set<std::vector<std::string>>> allowed(n);

  for (size_t i = 0; i < n; ++i) {
    size_t max_j = std::min(n, i + max_key_len);
    for (size_t j = i + 2; j <= max_j; ++j) {
      std::string key;
      std::string spaced;
      for (size_t k = i; k < j; ++k) {
        key += words[k];
        if (k != i) {
          spaced += " ";
        }
        spaced += words[k];
      }
      const auto &prons = lexicon.Get(key);
      const auto &spaced_prons = lexicon.Get(spaced);
      if (prons.empty() && spaced_prons.empty()) {
        continue;
      }
      for (const auto *set : {&prons, &spaced_prons}) {
        for (const auto &pron : *set) {
          if (pron.groups.size() != j - i) {
            continue;  // not split per token; unusable as a context
          }
          for (size_t k = 0; k < pron.groups.size(); ++k) {
            allowed[i + k].insert(pron.groups[k]);
          }
        }
      }
    }
  }

  std::vector<std::vector<std::vector<std::string>>> result(n);
  for (size_t i = 0; i < n; ++i) {
    const auto &base = lexicon.Get(words[i]);
    if (allowed[i].empty()) {
      if (base.empty()) {
        continue;  // out of vocabulary
      }
      for (const auto &p : base) {
        result[i].push_back(p.phones);
      }
    } else {
      // The context groups are valid readings of the token by construction;
      // they also cover tokens without a standalone entry.
      for (const auto &g : allowed[i]) {
        result[i].push_back(g);
      }
    }
  }
  return result;
}

fst::VectorFst<fst::StdArc> BuildFaLexiconFst(
    const std::vector<FaFstWord> &words, int32_t sil_id, float sil_prob) {
  if (!(sil_prob > 0.0f && sil_prob < 1.0f)) {
    throw std::runtime_error("silence_probability must be in (0, 1): " +
                             std::to_string(sil_prob));
  }

  float sil_cost = -std::log(sil_prob);
  float no_sil_cost = -std::log(1.0f - sil_prob);

  fst::VectorFst<fst::StdArc> fst;
  int32_t start_s = fst.AddState();
  int32_t loop = fst.AddState();
  int32_t sil_s = fst.AddState();
  fst.SetStart(start_s);
  fst.SetFinal(loop, fst::TropicalWeight::One());
  fst.AddArc(start_s,
             fst::StdArc(0, 0, fst::TropicalWeight(no_sil_cost), loop));
  fst.AddArc(start_s, fst::StdArc(0, 0, fst::TropicalWeight(sil_cost), sil_s));
  fst.AddArc(sil_s, fst::StdArc(sil_id, 0, fst::TropicalWeight::One(), loop));

  for (const auto &word : words) {
    for (const auto &pron : word.pronunciations) {
      if (pron.empty()) {
        continue;
      }
      int32_t cur = loop;
      for (size_t k = 0; k + 1 < pron.size(); ++k) {
        int32_t nxt = fst.AddState();
        // The word output label sits on the first phone of the path.
        fst.AddArc(cur, fst::StdArc(pron[k], k == 0 ? word.word_id : 0,
                                    fst::TropicalWeight::One(), nxt));
        cur = nxt;
      }
      int32_t last = pron.back();
      // Single-phone pronunciations carry the word label on their only arc.
      int32_t olab = (pron.size() == 1) ? word.word_id : 0;
      fst.AddArc(
          cur, fst::StdArc(last, olab, fst::TropicalWeight(no_sil_cost), loop));
      fst.AddArc(cur,
                 fst::StdArc(last, olab, fst::TropicalWeight(sil_cost), sil_s));
    }
  }

  fst::ArcSort(&fst, fst::OLabelCompare<fst::StdArc>());
  return fst;
}

}  // namespace sherpa_onnx
