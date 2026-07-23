#!/usr/bin/env python3
# Copyright    2026  (author: dora)
"""
Generate lexicon-ja.txt for kokoro-multi-lang-v1_0.

Requirements:
    pip install "misaki[ja]" wordfreq
    python -m unidic download

Note: kokoro-multi-lang-v1_0 was trained with phonemes produced by the
*cutlet* version of misaki's Japanese G2P (e.g. こんにちは -> koɲɲiʨiβa).
The newer pyopenjtalk-based G2P of misaki emits a pitch string
(characters like _ ^ -) and extra phonemes (e.g. ᶄ) that are NOT in
tokens.txt of v1.0, so we must use JAG2P(version='cutlet') here.

The lexicon contains:
  - the most frequent Japanese words (wordfreq) and their phonemes
  - every single kana (hiragana + katakana) and small-kana combination,
    so that any kana-only OOV text can still be synthesized
  - explicit entries for particles such as は/へ, mapped to their
    particle reading (wa/e). Longest-match lookup consumes real words
    first, so a remaining single は is most likely a topic particle.
"""

from typing import Dict, List

from misaki import ja
from wordfreq import top_n_list

NUM_WORDS = 100000
NUM_PARTICLE_WORDS = 20000

# Explicit overrides. Applied last, they win over generated entries.
MANUAL = {
    "は": "βa",  # topic particle
    "へ": "e",  # direction particle
    "を": "o",
    "っ": "ʔ",
    "ッ": "ʔ",
    "ー": "ː",
    "ん": "ɴ",
    "ン": "ɴ",
}

HIRAGANA_SMALL = "ゃゅょぁぃぅぇぉゎ"
KATAKANA_SMALL = "ャュョァィゥェォヮ"


def is_kana(ch: str) -> bool:
    return 0x3040 <= ord(ch) <= 0x309F or 0x30A0 <= ord(ch) <= 0x30FF


def is_japanese_word(w: str) -> bool:
    # kana or kanji only; skip latin/digits/mixed tokens
    for ch in w:
        if not (is_kana(ch) or 0x4E00 <= ord(ch) <= 0x9FFF or ch == "々"):
            return False
    return True


def kana_units() -> List[str]:
    """All single kana plus (kana + small kana) two-char units.

    Small kana are paired only with bases of the same script, so no
    cross-script combinations (e.g. あャ) are generated.
    """
    hiragana = [chr(c) for c in range(0x3041, 0x3097)]
    katakana = [chr(c) for c in range(0x30A1, 0x30FB)]
    units = hiragana + katakana
    units += [b + s for b in hiragana for s in HIRAGANA_SMALL]
    units += [b + s for b in katakana for s in KATAKANA_SMALL]
    return units


def main():
    g2p = ja.JAG2P(version="cutlet")

    def phonemize(w: str) -> str:
        ps, _ = g2p(w)
        ps = ps.replace(" ", "")
        if not ps or "❓" in ps:
            return ""
        return ps

    # dict keeps insertion order (Python 3.7+), deduplicates keys and
    # makes the manual overrides O(1)
    lexicon: Dict[str, str] = {}

    words = [w for w in top_n_list("ja", NUM_WORDS) if is_japanese_word(w)]

    for w in words:
        if w in lexicon:
            continue
        ps = phonemize(w)
        if ps:
            lexicon[w] = ps

    # Word + particle compounds (word+は, word+へ). The particle reading
    # (wa / e) is context-dependent; adding the compound as a longer
    # lexicon entry lets greedy longest-match resolve it correctly and
    # win over collisions such as 今日|はい|い (はい = "yes").
    for w in words[:NUM_PARTICLE_WORDS]:
        for p in ("は", "へ"):
            c = w + p
            if c in lexicon:
                continue
            ps = phonemize(c)
            if ps:
                lexicon[c] = ps

    for unit in kana_units():
        if unit in lexicon or unit in MANUAL:
            continue
        ps = phonemize(unit)
        if ps:
            lexicon[unit] = ps

    # Manual overrides win over generated entries.
    lexicon.update(MANUAL)

    with open("lexicon-ja.txt", "w", encoding="utf-8") as f:
        for word, phones in lexicon.items():
            tokens = " ".join(phones)
            f.write(f"{word} {tokens}\n")

    print(f"Wrote {len(lexicon)} entries to lexicon-ja.txt")


if __name__ == "__main__":
    main()
