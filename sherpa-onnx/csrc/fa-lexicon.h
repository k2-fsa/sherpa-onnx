// sherpa-onnx/csrc/fa-lexicon.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FA_LEXICON_H_
#define SHERPA_ONNX_CSRC_FA_LEXICON_H_

#include <stdint.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "fst/fstlib.h"

namespace sherpa_onnx {

// One pronunciation of a dictionary key.
//
// Lexicon lines are written as "key [probability columns ...] phones", where
// the phones of a multi-character key are split into one group per character
// with " || " (e.g. "研究 || i n ɛ n || tɕ i o u"). Single-character keys and
// plain (group-less) dictionaries have exactly one group.
struct FaPronunciation {
  // One phone group per character of the key.
  std::vector<std::vector<std::string>> groups;

  // All phones concatenated (what goes into the lexicon FST).
  std::vector<std::string> phones;
};

// Pronunciation dictionary: "key [probability columns ...] phones".
//
// Several lines under one key = several pronunciations of the same
// word/character; ALL of them are kept (identical phone sequences are
// deduplicated, file order is preserved) and each pronunciation becomes a
// parallel path in the lexicon FST, so the alignment picks the best one
// acoustically. This is how Chinese polyphones avoid text-level guessing.
struct FaLexicon {
  // key -> pronunciations
  std::unordered_map<std::string, std::vector<FaPronunciation>> entries;

  bool Contains(const std::string &key) const;

  // Returns the pronunciations of the key, owned by the lexicon. The
  // reference is empty when the key is not in the dictionary; it stays valid
  // as long as the lexicon is not modified.
  const std::vector<FaPronunciation> &Get(const std::string &key) const;

  // Parses the dictionary file. Numeric columns between the key and the
  // phones (optional probability columns) are ignored; " || " separates the
  // per-character phone groups. Throws std::runtime_error if the file cannot
  // be read or contains no valid entry.
  static FaLexicon Load(const std::string &filename);
};

// Tokenization for ForcedAligner::Compute() (no tokenize parameter):
//  - when the text is hanzi-based: drop whitespace/punctuation and emit one
//    token per character (the words tier is per character);
//  - otherwise: split on whitespace and strip punctuation from word edges.
std::vector<std::string> TokenizeFaText(const std::string &text,
                                        const FaLexicon &lexicon);

// Resolves the pronunciations of every token, with multi-token dictionary
// keys applied as pronunciation contexts: when consecutive tokens
// concatenate to a dictionary key (e.g. the hanzi pair 行长), the " || "
// phone groups of that key's pronunciations give the per-token readings
// directly (contextual polyphone pinning: 行长 -> hang zhang). Tokens without
// such a context keep all their own pronunciations. An empty list means the
// token is out of vocabulary.
//
// words must be the tokens returned by TokenizeFaText().
std::vector<std::vector<std::vector<std::string>>> ResolveFaPronunciations(
    const std::vector<std::string> &words, const FaLexicon &lexicon);

// One distinct word of an utterance, for BuildFaLexiconFst().
struct FaFstWord {
  int32_t word_id;  // output label of this word, >= 1
  // Phone-id sequences, one per pronunciation (parallel paths in the FST).
  // Must not be empty and each pronunciation must be non-empty.
  std::vector<std::vector<int32_t>> pronunciations;
};

// Builds the kaldi lexicon FST (L) for the given words:
//
//   start --(no silence)--> loop(final)
//   start --(silence cost)--> sil --(sil_id)--> loop
//   loop --(pronunciation path, word output label on its first phone)--> ...
//         --(last phone)--> loop (no silence) | sil (silence cost)
//
// with -log(sil_prob) / -log(1 - sil_prob) as the silence costs. The result
// is olabel-sorted (as the TrainingGraphCompiler expects).
fst::VectorFst<fst::StdArc> BuildFaLexiconFst(
    const std::vector<FaFstWord> &words, int32_t sil_id, float sil_prob);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FA_LEXICON_H_
