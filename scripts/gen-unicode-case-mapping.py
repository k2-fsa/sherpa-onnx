#!/usr/bin/env python3
# Regenerate sherpa-onnx/csrc/unicode-case-mapping.{h,cc}.
#
# Python's `unicodedata` is the source of truth, so no UnicodeData.txt
# download is needed. Produces three tables:
#   * simple (1:1) lowercase mappings for every cased letter
#   * codepoint ranges for Unicode Letter + Mark, used to strip/split
#     punctuation away from OOV lexicon tokens
#   * canonical decompositions, combining classes and primary composites,
#     used to normalize text to Unicode NFC before lexicon lookup
#
# Usage:  python3 scripts/gen-unicode-case-mapping.py

import os
import sys
import unicodedata


def nfd_expand(cp: int, depth: int = 0) -> list:
    """Full recursive canonical decomposition of |cp|."""
    d = unicodedata.decomposition(chr(cp))
    if not d or d.startswith("<") or depth > 4:
        return [cp]
    out = []
    for p in (int(x, 16) for x in d.split()):
        out.extend(nfd_expand(p, depth + 1))
    return out


def fmt_rows(items, per):
    out = []
    for i in range(0, len(items), per):
        chunk = " ".join(
            "{" + ",".join(str(x) for x in it) + "}," for it in items[i : i + per]
        )
        out.append("    " + chunk)
    return "\n".join(out)


HEADER = """// sherpa-onnx/csrc/unicode-case-mapping.h
//
// GENERATED FILE - DO NOT EDIT BY HAND.
//
// Regenerate with scripts/gen-unicode-case-mapping.py (which reads Python's
// `unicodedata`). These tables replace the hand-maintained switch in
// ToLowerCase() that could only cover a few dozen characters and silently
// left the rest (e.g. U+0100 LATIN CAPITAL LETTER A WITH MACRON) uppercase
// because std::towlower() is a no-op for non-ASCII in the "C" locale.
//
// This file is part of sherpa-onnx.
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_UNICODE_CASE_MAPPING_H_
#define SHERPA_ONNX_CSRC_UNICODE_CASE_MAPPING_H_

#include <cstdint>
#include <string>

namespace sherpa_onnx {

// Simple (1:1) Unicode lowercase mapping. Returns cp unchanged when there is
// no single-codepoint lowercase form. Not locale-dependent, unlike
// std::towlower().
char32_t ToSimpleLower(char32_t cp);

// True if cp is a Unicode Letter (category L*) or Mark (category M*), i.e.
// something that can appear inside a lexicon word. Used to strip/split
// punctuation away from OOV tokens before retrying the lexicon.
bool IsLexiconWordChar(char32_t cp);

// Canonical combining class of |cp| (0 for starters).
uint32_t GetCanonicalCombiningClass(char32_t cp);

// In-place Unicode NFC (canonical decomposition, canonical ordering,
// canonical composition). Needed so that e.g. "a" + U+0301 matches a lexicon
// entry stored as U+00E1. Returns true if |s| changed.
bool NormalizeNfc(std::u32string *s);

// Convenience overload for UTF-8 input.
std::string NormalizeNfc(const std::string &s);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_UNICODE_CASE_MAPPING_H_
"""

CC_HEAD = """// sherpa-onnx/csrc/unicode-case-mapping.cc
//
// GENERATED FILE - DO NOT EDIT BY HAND. See unicode-case-mapping.h.
//
// This file is part of sherpa-onnx.
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/unicode-case-mapping.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

struct CodepointPair {
  char32_t from;
  char32_t to;
};

struct CodepointTriple {
  char32_t a;
  char32_t b;
  char32_t c;
};

// {cp, len, d0, d1, d2, d3} -- full canonical decomposition of |cp| into at
// most 4 codepoints. Sorted by |cp|.
struct Decomposition {
  char32_t cp;
  int32_t len;
  char32_t d[4];
};

"""

CC_TAIL = """
uint32_t GetCanonicalCombiningClass(char32_t cp) {
  auto it = std::lower_bound(
      kCombiningClass, kCombiningClass + kCombiningClassSize, cp,
      [](const CodepointPair &p, char32_t v) { return p.from < v; });
  if (it != kCombiningClass + kCombiningClassSize && it->from == cp) {
    return it->to;
  }
  return 0;
}

namespace {

// Hangul syllable composition (UAX #15 section 16). Algorithmic, so the
// syllables are absent from both tables above.
constexpr char32_t kHangulSBase = 0xAC00;
constexpr char32_t kHangulLBase = 0x1100;
constexpr char32_t kHangulVBase = 0x1161;
constexpr char32_t kHangulTBase = 0x11A7;
constexpr int32_t kHangulLCount = 19;
constexpr int32_t kHangulVCount = 21;
constexpr int32_t kHangulTCount = 28;
constexpr int32_t kHangulNCount = kHangulVCount * kHangulTCount;  // 588
constexpr int32_t kHangulSCount = kHangulLCount * kHangulNCount;  // 11172

// Recursive canonical decomposition of |cp| into |out|.
void Decompose(char32_t cp, std::u32string *out) {
  // Hangul syllables decompose algorithmically and so are absent from
  // kDecompose.
  if (cp >= kHangulSBase && cp < kHangulSBase + kHangulSCount) {
    int32_t sindex = static_cast<int32_t>(cp - kHangulSBase);
    out->push_back(kHangulLBase + sindex / kHangulNCount);
    out->push_back(kHangulVBase + (sindex % kHangulNCount) / kHangulTCount);
    int32_t t = sindex % kHangulTCount;
    if (t != 0) out->push_back(kHangulTBase + t);
    return;
  }

  auto it = std::lower_bound(
      kDecompose, kDecompose + kDecomposeSize, cp,
      [](const Decomposition &p, char32_t v) { return p.cp < v; });
  if (it == kDecompose + kDecomposeSize || it->cp != cp) {
    out->push_back(cp);
    return;
  }
  for (int32_t i = 0; i < it->len; ++i) Decompose(it->d[i], out);
}

// Canonical composition of (first, second) -> composite.
bool TryCompose(char32_t first, char32_t second, char32_t *composed) {
  // L + V -> LV, and LV + T -> LVT.
  if (first >= kHangulLBase && first < kHangulLBase + kHangulLCount &&
      second >= kHangulVBase && second < kHangulVBase + kHangulVCount) {
    *composed = kHangulSBase +
                ((first - kHangulLBase) * kHangulVCount +
                 (second - kHangulVBase)) *
                    kHangulTCount;
    return true;
  }
  if (first >= kHangulSBase && first < kHangulSBase + kHangulSCount &&
      (first - kHangulSBase) % kHangulTCount == 0 &&
      second > kHangulTBase && second < kHangulTBase + kHangulTCount) {
    *composed = first + (second - kHangulTBase);
    return true;
  }

  auto it = std::lower_bound(
      kCompose, kCompose + kComposeSize, CodepointTriple{first, second, 0},
      [](const CodepointTriple &p, const CodepointTriple &v) {
        return p.a != v.a ? p.a < v.a : p.b < v.b;
      });
  if (it != kCompose + kComposeSize && it->a == first && it->b == second) {
    *composed = it->c;
    return true;
  }
  return false;
}

}  // namespace

bool NormalizeNfc(std::u32string *s) {
  if (s->empty()) return false;

  // 1. Canonical decomposition.
  std::u32string nfd;
  nfd.reserve(s->size());
  for (char32_t cp : *s) Decompose(cp, &nfd);

  // 2. Canonical ordering: a stable bubble sort of each non-starter run by
  //    combining class (UAX #15 D108).
  for (std::size_t i = 1; i < nfd.size(); ++i) {
    uint32_t cc = GetCanonicalCombiningClass(nfd[i]);
    if (cc == 0) continue;
    for (std::size_t j = i; j > 0; --j) {
      uint32_t prev = GetCanonicalCombiningClass(nfd[j - 1]);
      if (prev == 0 || prev <= cc) break;
      std::swap(nfd[j - 1], nfd[j]);
    }
  }

  // 3. Canonical composition (UAX #15 D117 blocking rule).
  std::u32string out;
  out.reserve(nfd.size());
  for (char32_t cp : nfd) {
    if (out.empty()) {
      out.push_back(cp);
      continue;
    }

    uint32_t cc_cp = GetCanonicalCombiningClass(cp);
    if (cc_cp == 0) {
      // A starter may compose with the starter directly before it; anything
      // intervening blocks it (UAX #15 D117). Bengali U+09CB, Tamil U+0BCA and
      // friends are two starters that must recompose, so this must not be a
      // blanket early return.
      char32_t composed = 0;
      if (GetCanonicalCombiningClass(out.back()) == 0 &&
          TryCompose(out.back(), cp, &composed)) {
        out.back() = composed;
      } else {
        out.push_back(cp);
      }
      continue;
    }

    std::size_t starter = out.size() - 1;
    while (starter > 0 && GetCanonicalCombiningClass(out[starter]) != 0) {
      --starter;
    }
    if (GetCanonicalCombiningClass(out[starter]) != 0) {
      out.push_back(cp);
      continue;
    }

    // C is blocked from S when some B between them has ccc(B) >= ccc(C).
    bool blocked = false;
    for (std::size_t k = starter + 1; k < out.size(); ++k) {
      if (GetCanonicalCombiningClass(out[k]) >= cc_cp) {
        blocked = true;
        break;
      }
    }
    if (blocked) {
      out.push_back(cp);
      continue;
    }

    char32_t composed = 0;
    if (TryCompose(out[starter], cp, &composed)) {
      out[starter] = composed;
    } else {
      out.push_back(cp);
    }
  }

  if (out == *s) return false;
  *s = std::move(out);
  return true;
}

std::string NormalizeNfc(const std::string &s) {
  std::u32string u32 = Utf8ToUtf32(s);
  NormalizeNfc(&u32);
  return Utf32ToUtf8(u32);
}

}  // namespace sherpa_onnx
"""


def main() -> int:
    pairs, word_cps, dec, ccc, comp = [], [], [], [], []

    for cp in range(0x110000):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        # L = letter, M = combining mark, N = number. Numbers must count as
        # word characters or pinyin keys like "zhong1-guo2" lose their tone
        # digits when a punctuation-bearing token is split.
        if cat.startswith(("L", "M", "N")):
            word_cps.append(cp)

        # Simple (1:1) lowercase mapping. Python's str.lower() is the *full*
        # lowercase mapping and may expand -- U+0130 gives i + U+0307, U+00DF
        # gives ss -- so taking it verbatim misses U+0130 -> i and would be
        # wrong for U+00DF. The rule: a one-codepoint result is the mapping;
        # a result that is one base plus combining marks has that base as its
        # simple mapping; anything else has none. Do not restrict by category,
        # or cased numerals (U+2167) and circled letters (U+24B6) are lost.
        lo = ch.lower()
        if len(lo) == 1:
            mapped = ord(lo)
        elif lo and all(unicodedata.category(c).startswith("M") for c in lo[1:]):
            # One base plus combining marks: the base is the simple mapping.
            # Only the tail may be marks -- the base itself is a letter.
            mapped = ord(lo[0])
        else:
            mapped = cp
        if mapped != cp:
            pairs.append((cp, mapped))

        d = unicodedata.decomposition(ch)
        c = unicodedata.combining(ch)
        if c:
            ccc.append((cp, c))
        if d and not d.startswith("<"):
            full = nfd_expand(cp)
            if full != [cp]:
                padded = list(full) + [0] * (4 - len(full))
                dec.append((cp, len(full), padded[0], padded[1], padded[2], padded[3]))
            if len(full) >= 2:
                # Primary composites must come from the *immediate* canonical
                # decomposition, and only when NFC really recomposes to |cp|.
                # Guarding on the fully-expanded length is wrong: multi-stage
                # pairs such as â + acute -> ấ expand to three codepoints but
                # still have a two-part immediate decomposition. The NFC check
                # drops composition-exclusion pairs such as क + nukta -> U+0958
                # that NFC deliberately refuses to compose.
                imm = [int(x, 16) for x in d.split()]
                if len(imm) == 2:
                    base, mark = imm
                    recomposed = unicodedata.normalize("NFC", chr(base) + chr(mark))
                    if len(recomposed) == 1 and ord(recomposed) == cp:
                        comp.append((base, mark, cp))

    # kCompose is binary-searched by (base, mark), so it must be sorted by
    # that key -- not by the composite codepoint the loop above walks.
    comp.sort(key=lambda t: (t[0], t[1]))

    intervals = []
    for cp in word_cps:
        if intervals and intervals[-1][1] == cp - 1:
            intervals[-1][1] = cp
        else:
            intervals.append([cp, cp])

    body = []
    body.append("// Sorted by |from|; simple lowercase mappings for every cased letter.")
    body.append("constexpr CodepointPair kToLower[] = {")
    body.append(fmt_rows(pairs, 4))
    body.append("};")
    body.append("")
    body.append("// Sorted, non-overlapping [from, to] ranges of Unicode Letter + Mark.")
    body.append("constexpr CodepointPair kWordCharRanges[] = {")
    body.append(fmt_rows(intervals, 4))
    body.append("};")
    body.append("")
    body.append("// Canonical combining classes (non-zero only), sorted by |from|.")
    body.append("constexpr CodepointPair kCombiningClass[] = {")
    body.append(fmt_rows(ccc, 4))
    body.append("};")
    body.append("")
    body.append("// Full canonical decompositions, sorted by |cp|.")
    body.append("constexpr Decomposition kDecompose[] = {")
    body.append(fmt_rows(dec, 3))
    body.append("};")
    body.append("")
    body.append("// Primary composites, sorted by (a, b).")
    body.append("constexpr CodepointTriple kCompose[] = {")
    body.append(fmt_rows(comp, 3))
    body.append("};")
    body.append("")
    for name, arr in (
        ("kToLowerSize", "kToLower"),
        ("kWordCharRangesSize", "kWordCharRanges"),
        ("kCombiningClassSize", "kCombiningClass"),
        ("kDecomposeSize", "kDecompose"),
        ("kComposeSize", "kCompose"),
    ):
        body.append(f"constexpr std::size_t {name} = sizeof({arr}) / sizeof({arr}[0]);")
    body.append("")
    body.append("}  // namespace")
    body.append("")
    body.append(
        "char32_t ToSimpleLower(char32_t cp) {\n"
        "  auto it = std::lower_bound(\n"
        "      kToLower, kToLower + kToLowerSize, cp,\n"
        "      [](const CodepointPair &p, char32_t v) { return p.from < v; });\n"
        "  if (it != kToLower + kToLowerSize && it->from == cp) {\n"
        "    return it->to;\n"
        "  }\n"
        "  return cp;\n"
        "}\n"
        "\n"
        "bool IsLexiconWordChar(char32_t cp) {\n"
        "  auto it = std::lower_bound(\n"
        "      kWordCharRanges, kWordCharRanges + kWordCharRangesSize, cp,\n"
        "      [](const CodepointPair &p, char32_t v) { return p.to < v; });\n"
        "  return it != kWordCharRanges + kWordCharRangesSize && cp >= it->from &&\n"
        "         cp <= it->to;\n"
        "}"
    )
    body.append("")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hdr_path = os.path.join(root, "sherpa-onnx", "csrc", "unicode-case-mapping.h")
    cc_path = os.path.join(root, "sherpa-onnx", "csrc", "unicode-case-mapping.cc")

    with open(hdr_path, "w") as f:
        f.write(HEADER)
    with open(cc_path, "w") as f:
        f.write(CC_HEAD + "\n".join(body) + CC_TAIL)

    print(hdr_path)
    print(cc_path)
    print(
        f"  lowercase={len(pairs)} wordchar={len(intervals)} ccc={len(ccc)} "
        f"decompose={len(dec)} compose={len(comp)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
