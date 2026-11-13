#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corporation
#
# See ../../../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate lexicon.txt (word -> phones) for this English MFA model.

The word list is enumerated exhaustively from lexicon-en-us.txt: only its
words and the words of its phrases are used; the pronunciations in that file
are ignored. Pronunciations come from this model's MFA dictionary by
exhaustive lookup; a word without a dictionary pronunciation gets no
lexicon entry, and the aligner reports it as out-of-vocabulary at runtime.

Each line of the output is one pronunciation:

  word phone1 phone2 ...

Multiple lines under one word = multiple pronunciations; the aligner keeps
them all as parallel paths and the acoustics pick the best one.

Usage (from this directory, or via ./run.sh):

  python3 generate_lexicon.py [--model-dir DIR] [--out lexicon.txt]

The download/ directory must contain the acoustic model (meta.json +
phones.txt), the MFA dictionary (*.dict) and lexicon-en-us.txt.

References (sources this implementation is based on):
- Pronunciations: the MFA dictionaries of
  https://github.com/MontrealCorpusTools/mfa-models
- Word list: https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
"""

import argparse
import json
import unicodedata
from pathlib import Path


def candidate_dirs(script_dir):
    return [script_dir, script_dir / "download"] + sorted(
        script_dir.glob("download/*")
    ) + sorted(script_dir.glob("download/*/*"))


def find_model_dir(script_dir):
    """The acoustic model directory: the one holding meta.json."""
    for cand in candidate_dirs(script_dir):
        if (cand / "meta.json").is_file():
            return cand
    raise SystemExit(
        "error: no acoustic model directory (meta.json) found; "
        "run ./run.sh first"
    )


def find_dictionary(script_dir):
    """The MFA dictionary (*.dict) shipped by run.sh."""
    for cand in candidate_dirs(script_dir):
        dicts = sorted(cand.glob("*.dict"))
        if dicts:
            return dicts[0]
    raise SystemExit("error: no MFA dictionary (*.dict) found; run ./run.sh first")


def find_word_list(script_dir):
    for cand in [script_dir, script_dir / "download"]:
        p = cand / "lexicon-en-us.txt"
        if p.is_file():
            return p
    raise SystemExit("error: lexicon-en-us.txt not found; run ./run.sh first")


def load_word_list(path):
    """Word list from lexicon-en-us.txt: single words (lowercased,
    deduplicated) plus the multi-word phrases with their combined
    pronunciations (stress marks stripped). Returns (words, seen, phrases)
    where phrases is [(phrase words, [pron, ...]), ...]."""
    seen = set()
    words = []
    phrases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            key, phones = line.rstrip("\n").split(" || ")
            pron = tuple(clean_phones(phones.split()))
            if " " in key:
                norm_key = " ".join(
                    unicodedata.normalize("NFKD", w).lower() for w in key.split()
                )
                phrases.append((norm_key, pron))
                continue
            word = unicodedata.normalize("NFKD", key).lower()
            if word and word not in seen:
                seen.add(word)
                words.append(word)
    # collapse duplicate phrase pronunciations
    merged = {}
    for key, pron in phrases:
        merged.setdefault(key, [])
        if pron not in merged[key]:
            merged[key].append(pron)
    phrases = list(merged.items())
    return words, seen, phrases


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



def split_phrase_pron(combined, words, single):
    """Split a phrase pronunciation into one phone group per word.

    Prefers segments that are standalone pronunciations of the word; the
    residual segments are the combined/special readings of the phrase
    (like polyphones). Returns None when no split is found.
    """
    best = None

    def dfs(ti, pos, cur, matched):
        nonlocal best
        if ti == len(words) - 1:
            seg = combined[pos:]
            if not seg:
                return
            score = matched + (seg in single.get(words[ti], ()))
            cand = (score, cur + [list(seg)])
            if best is None or cand[0] > best[0]:
                best = cand
            return
        if pos >= len(combined):
            return
        rest = len(words) - ti
        max_len = len(combined) - pos - (rest - 1)
        # prefer a standalone pronunciation of this word
        for pron in single.get(words[ti], ()):
            n = len(pron)
            if n <= max_len and tuple(combined[pos : pos + n]) == pron:
                dfs(ti + 1, pos + n, cur + [list(pron)], matched + 1)
        # otherwise the word has a special reading: try any span
        for n in range(1, max_len + 1):
            seg = tuple(combined[pos : pos + n])
            if seg in single.get(words[ti], ()):
                continue
            dfs(ti + 1, pos + n, cur + [list(seg)], matched)

    dfs(0, 0, [], 0)
    return best[1] if best else None


# The word list writes a few phones with different conventions than the MFA
# phone set; map them onto the MFA spelling.
PHONE_MAP = {
    "r": "\u0279",     # r -> ɹ
    "\u028c": "\u0250",   # ʌ -> ɐ (STRUT)
    "\u0254\u02d0": "\u0252\u02d0",  # ɔː -> ɒː (THOUGHT)
    "\u1d7b": "\u026a",   # ᵻ -> ɪ
    "\u0254\u02d0\u02d0": "\u0252\u02d0",  # ɔːː -> ɒː
}


def clean_phones(tokens):
    """Strip stress marks and map phones onto the MFA spelling."""
    out = []
    for p in tokens:
        p = p.replace("\u02c8", "").replace("\u02cc", "")
        if not p or p == ".":
            continue
        out.append(PHONE_MAP.get(p, p))
    return out

def load_dictionary(path):
    """word -> list of pronunciations (one per line); lowercased keys."""
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            word = unicodedata.normalize("NFKD", parts[0]).lower()
            if word.startswith("<") or word.startswith("["):
                continue  # special entries (<unk>, [bracketed], ...)
            # optional probability columns between the key and the phones;
            # stress marks may be attached to a phone or standalone
            phones = tuple(clean_phones(p for p in parts[1:] if not _is_float(p)))
            if not phones:
                continue
            bucket = entries.setdefault(word, [])
            if phones not in bucket:
                bucket.append(phones)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="",
                        help="acoustic model dir (phones.txt is used for "
                        "validation); auto-detected by default")
    parser.add_argument("--out", default="",
                        help="output path; lexicon.txt by default")
    args = parser.parse_args()

    script_dir = Path(__file__).absolute().parent
    model_dir = Path(args.model_dir) if args.model_dir else find_model_dir(script_dir)
    dict_path = find_dictionary(script_dir)
    word_list_path = find_word_list(script_dir)
    out_path = Path(args.out) if args.out else script_dir / "lexicon.txt"

    # phones.txt is the phone table; older models only list the phones in
    # meta.json.
    phones = set()
    phones_txt = model_dir / "phones.txt"
    if phones_txt.is_file():
        with open(phones_txt, encoding="utf-8") as f:
            for line in f:
                phones.add(line.split()[0])
    else:
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        phones.update(meta.get("phones", []))

    words, seen, phrases = load_word_list(word_list_path)
    dictionary = load_dictionary(dict_path)
    single = {w: set(prons) for w, prons in dictionary.items()}

    # Cover as many words as possible: the word list is the union of the
    # provided words and the dictionary's own words.
    for word in dictionary:
        if word not in seen:
            seen.add(word)
            words.append(word)

    n_lines = 0
    n_words = 0
    n_oov = 0
    n_phrases = 0
    n_phrase_skipped = 0
    unknown = set()
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for word in words:
            prons = dictionary.get(word)
            if not prons:
                n_oov += 1
                continue
            written = 0
            for pron in prons:
                if any(p not in phones for p in pron):
                    unknown.update(p for p in pron if p not in phones)
                    continue
                f.write(word + " " + " ".join(pron) + "\n")
                written += 1
                n_lines += 1
            if written:
                n_words += 1
            else:
                n_oov += 1

    # Phrase entries: the combined readings of multi-word phrases (they
    # differ from the concatenation of the word readings, like polyphones).
    # Their pronunciations come from lexicon-en-us.txt; the per-word phone
    # groups come from splitting the combined pronunciation.
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for phrase, prons in phrases:
            written = 0
            for pron in prons:
                groups = split_phrase_pron(pron, phrase.split(), single)
                if not groups:
                    continue
                flat = [p for g in groups for p in g]
                if any(p not in phones for p in flat):
                    unknown.update(p for p in flat if p not in phones)
                    continue
                f.write(
                    phrase + " || " + " || ".join(" ".join(g) for g in groups)
                    + "\n"
                )
                written += 1
                n_lines += 1
            if written:
                n_phrases += 1
            else:
                n_phrase_skipped += 1

    print(f"model dir:  {model_dir}")
    print(f"dictionary: {dict_path} ({len(dictionary)} entries)")
    print(f"word list:  {word_list_path} ({len(words)} words)")
    print(f"lexicon:    {n_lines} lines, {n_words} words, {n_phrases} phrases")
    if n_oov:
        print(
            f"warning: {n_oov} words have no dictionary pronunciation and are "
            f"not in the lexicon; the aligner reports OOV at runtime"
        )
    if n_phrase_skipped:
        print(f"warning: {n_phrase_skipped} phrases skipped (no usable pronunciation)")
    if unknown:
        print(
            f"warning: {len(unknown)} dictionary phones are not in the model's "
            f"phones.txt: {' '.join(sorted(unknown))}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
