#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# See ../../../../../LICENSE for clarification regarding multiple authors
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

"""Generate lexicon.txt (hanzi/phrase -> phones) for the lars76 MFA
forced-alignment models (aishell3_pinyin / biaobei_pinyin / large_pinyin,
https://github.com/lars76/forced-alignment-chinese/releases/tag/model_files).

This script is shared by the three model directories: their
generate_lexicon.py entries are symlinks to this file, and the model to use
is auto-detected under the directory the symlink lives in.

The lexicon maps Chinese characters and phrases directly to the phone
sequences expected by this model's phone set, so a forced aligner needs a
single dictionary (no runtime pypinyin, no pinyin-keyed dictionary):

  行 ɕ i ŋ        <- polyphone: one line per reading, all parallel
  行 x a ŋ           (hang -> x a ŋ / h a ŋ free variants; heng likewise)
  行 h a ŋ
  银行 i n || x a ŋ   <- phrase: " || " marks the per-character phone groups

- Single characters: every reading of a polyphone becomes its own line; the
  aligner keeps the lines as parallel pronunciations and the acoustics pick
  the right one (no guessing at text-processing time).
- Phrases: readings come from pypinyin's phrase dictionary with tone sandhi
  (行长 -> hang zhang), which pins polyphone readings in context; the " || "
  groups tell the aligner which phones belong to which character, so its
  words tier is per character.
- Erhua: for a key ending in 儿 (e.g. 花儿, 一点儿) an extra merged
  pronunciation (e.g. huar, dianr) is added alongside the plain one; its
  last phone (the rhotic tail) is grouped with 儿.

The pinyin-syllable -> IPA phones come from this model's own pronunciation
dictionary (the `*_dictionary.txt` file shipped in the release bundle),
falling back to the pinyin_to_ipa conversion for syllables the dictionary
does not cover. The conversion (including erhua) is ported from
create_dictionary.py of https://github.com/lars76/forced-alignment-chinese
); its phones are fitted to this model's phones.txt inventory (combined
erhua phones like ɐʵ are split into ɐ ɻ when the model was trained that way).
The conversion was verified to reproduce the upstream aishell3_pinyin/
large_pinyin training dictionaries exactly.

Usage (from this directory, or via ./run.sh):

  python3 -I generate_lexicon.py [--model-dir DIR] [--out lexicon.txt]

Output: lexicon.txt, one pronunciation per line,
"key phone1 ... || phone1 ... || ...", no probability columns (" || "
separates the per-character phone groups). Requires:
pip install pypinyin pinyin_to_ipa

References (sources this implementation is based on):
- Pinyin -> IPA conversion (incl. the erhua rules, the ERHUA_SUFFIX_TO_IPA
  table and the phone-splitting fallback) is ported from create_dictionary.py
  of https://github.com/lars76/forced-alignment-chinese,
  which generated the pronunciation dictionaries the lars76 MFA models were
  trained on.
- Character/phrase enumeration with pypinyin (pinyin_dict + phrases_dict,
  polyphone readings) follows generate_lexicon.py of
  https://github.com/k2-fsa/icefall/blob/master/egs/baker_zh/TTS/matcha/
  (Apache-2.0 license).
- Character/phrase readings: https://github.com/mozillazg/python-pinyin
- Pinyin -> IPA tables: https://github.com/clem109/pinyin-to-ipa
"""

import argparse
import itertools
import re
from pathlib import Path

from pinyin_to_ipa import pinyin_to_ipa
from pypinyin import Style, lazy_pinyin, phrases_dict, pinyin, pinyin_dict

# ---------------------------------------------------------------------------
# Pinyin syllable -> IPA phones.
# Ported from create_dictionary.py of
# https://github.com/lars76/forced-alignment-chinese, which generated
# the pronunciation dictionaries the lars76 MFA models were trained on.
# ---------------------------------------------------------------------------

# Dictionary for erhua transformations
# https://en.wikipedia.org/wiki/Erhua#Standard_rules
ERHUA_SUFFIX_TO_IPA = {
    "uangr": [["w", "ɑ̃ʵ"]],
    "iangr": [["j", "ɑ̃ʵ"]],
    "iongr": [["j", "ʊ̃ʵ"]],
    "vanr": [["ɥ", "ɐʵ"]],
    "uair": [["w", "ɐʵ"]],
    "ianr": [["j", "ɐʵ"]],
    "iaor": [["j", "ɑu̯˞"]],
    "uanr": [["w", "ɐʵ"]],
    "engr": [["ɤ̃ʵ"]],
    "angr": [["ɑ̃ʵ"]],
    "ongr": [["w", "ɤ̃ʵ"], ["ʊ̃˞"]],
    "ingr": [["j", "ɤ̃ʵ"]],
    "ver": [["ɥ", "œʵ"]],
    "uar": [["w", "äʵ"], ["w", "ɐʵ"]],
    "uor": [["w", "ɔʵ"]],
    "air": [["ɐʵ"]],
    "eir": [["ɚ"]],
    "aor": [["ɑu̯˞"]],
    "our": [["ou̯˞"]],
    "anr": [["ɐʵ"]],
    "enr": [["ɚ"]],
    "iar": [["j", "äʵ"], ["j", "ɐʵ"]],
    "ier": [["j", "ɛʵ"]],
    "iur": [["j", "ou̯ʵ"]],
    "inr": [["j", "ɚ"]],
    "uir": [["w", "ɚ"]],
    "unr": [["w", "ɚ"]],
    "vnr": [["ɥ", "ɚ"]],
    "ar": [["äʵ"], ["ɐʵ"]],
    "or": [["ɔʵ"]],
    "er": [["ɤʵ"]],
    "ur": [["u˞"]],
    "vr": [["ɥ", "ɚ"]],
    "ir": [["ɚ"]],
}

# If a converted phone is not in the model's phones.txt, try these string
# replacements (create_dictionary.py's SPLIT_PHONEMES): some models were
# trained with combined erhua phones (ɐʵ), others with them split (ɐ ɻ).
SPLIT_PHONEMES = [("ʵ", " ɻ"), ("˞", " ɻ"), ("ɚ", "ə ɻ"), ("aə", "a")]


def apply_erhua(pinyin_syl, ipa):
    """Apply erhua transformation to the given IPA representation."""
    # Remove final 'n' or 'ŋ' if present
    if ipa[-1] in ["ŋ", "n"]:
        ipa = ipa[:-1]

    # Remove last character if IPA has more than one character
    if len(ipa) > 1:
        ipa = ipa[:-1]

    result = []
    for pinyin_ending, ipa_endings in ERHUA_SUFFIX_TO_IPA.items():
        if pinyin_syl.endswith(pinyin_ending) and pinyin_syl != pinyin_ending:
            for ipa_ending in ipa_endings:
                if ipa[-1] == ipa_ending[0]:
                    ipa_ending = ipa_ending[1:]
                elif ipa[-1] + "˞" == ipa_ending[0]:  # wur
                    ipa = []
                result.append(ipa + ipa_ending)
            break
    return result


def convert_pinyin_to_ipa(pinyin_syl):
    """Convert a toneless Pinyin syllable to its IPA phone sequence(s)."""
    # Check for erhua
    erhua = pinyin_syl.endswith("r") and pinyin_syl != "er"
    if erhua:
        pinyin_syl = pinyin_syl[:-1]

    ipa_output = []
    for ipa in pinyin_to_ipa(pinyin_syl):
        ipa = list(ipa)

        # Special case for "yo"
        if pinyin_syl == "yo" and ipa == ["w", "o"]:
            ipa = ["j", "ɔ"]

        if erhua:
            erhua_ipas = apply_erhua(pinyin_syl + "r", ipa)
            ipa_output.extend([i for i in erhua_ipas if i not in ipa_output])
        else:
            ipa_output.append(ipa)

    return ipa_output


def fit_phones(ipa, inventory):
    """Fit converted phones to the model's phone inventory.

    Returns the phone list with combined erhua phones split when the model
    does not have them, or None if some phone cannot be fitted.
    """
    out = []
    for p in ipa:
        if p in inventory:
            out.append(p)
            continue
        for s0, s1 in SPLIT_PHONEMES:
            if s0 in p:
                cand = p.replace(s0, s1).split()
                if all(t in inventory for t in cand):
                    out.extend(cand)
                    break
        else:
            return None
    return out


def _is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def load_pinyin_dictionary(path):
    """Parse an MFA pinyin pronunciation dictionary.

    Format (the `*_dictionary.txt` file from the release bundle):
    "key [probability columns ...] phone1 phone2 ..."; one key may have
    several lines = several pronunciations (free variation / alternates).
    Returns {toneless pinyin key: [phone tuple, ...]}.
    """
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            phones = tuple(t for t in parts[1:] if not _is_float(t))
            if not phones:
                continue
            bucket = table.setdefault(parts[0], [])
            if phones not in bucket:
                bucket.append(phones)
    return table


def find_pinyin_dictionary(script_dir, model_dir):
    """Find this model's own pinyin dictionary (optional but authoritative)."""
    for base in (script_dir, script_dir / "download", model_dir, model_dir.parent):
        for pattern in ("*_dictionary.txt", "*_dictionary.dict"):
            for cand in sorted(base.glob(pattern)):
                return cand
    return None


class SyllableMap:
    """Toneless pinyin syllable -> list of phone-tuple pronunciations."""

    def __init__(self, inventory, table=None):
        self.inventory = inventory
        self.table = table or {}
        self.cache = {}
        self.failed = set()
        self.n_from_table = 0
        self.n_from_conversion = 0

    def pronunciations(self, syl):
        if syl in self.cache:
            return self.cache[syl]
        out = []
        if syl in self.table:
            out = list(self.table[syl])
            self.n_from_table += 1
        else:
            try:
                ipas = convert_pinyin_to_ipa(syl)
            except Exception:
                ipas = []
            for ipa in ipas:
                phones = fit_phones(ipa, self.inventory)
                if phones is None or not phones:
                    continue
                t = tuple(phones)
                if t not in out:
                    out.append(t)
            if out:
                self.n_from_conversion += 1
        if not out:
            self.failed.add(syl)
        self.cache[syl] = out
        return out


def strip_tone(syl):
    return re.sub(r"\d+$", "", syl).lower()


def char_readings(ch):
    """All toneless readings of one character (polyphones -> several)."""
    out = []
    for p in pinyin(ch, heteronym=True, style=Style.TONE3)[0]:
        s = strip_tone(p)
        if s not in out:
            out.append(s)
    return out


def phrase_readings(word):
    """Contextual (tone-sandhi) reading of a phrase, one syllable per char."""
    return [strip_tone(p) for p in lazy_pinyin(word, style=Style.TONE3, tone_sandhi=True)]


def erhua_variant(word, syllables):
    """Merged erhua pronunciation of a word ending in 儿, e.g. 一点儿 -> yi dianr.

    Returns the merged syllable list, or None when the word is not an erhua
    candidate (儿 not word-final, or a word like 虐儿/二儿).
    """
    if len(word) < 2 or word[-1] != "儿" or word[-2] in "虐二":
        return None
    if len(syllables) != len(word) or syllables[-1] != "er":
        return None
    prev = syllables[-2]
    if prev.endswith("r"):
        return None
    return syllables[:-2] + [prev + "r"]


def find_model_dir(script_dir):
    candidates = [script_dir]
    candidates += sorted(script_dir.glob("download/*"))
    candidates += sorted(p for p in script_dir.glob("*") if p.is_dir())
    for cand in candidates:
        if (cand / "meta.json").is_file() and (cand / "phones.txt").is_file():
            return cand
    raise SystemExit(
        "error: no MFA acoustic model directory (meta.json + phones.txt) found "
        f"under {script_dir}. Run ./run.sh to download the model, or pass "
        "--model-dir."
    )


def load_inventory(model_dir):
    inventory = set()
    with open(model_dir / "phones.txt", encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if fields:
                inventory.add(fields[0])
    return inventory


# Cap on the pronunciations emitted per reading (some syllables have several
# IPA transcriptions in free variation, e.g. hua -> h w a / x w a; the product
# over a phrase is capped so long phrases cannot explode).
MAX_PRONUNCIATIONS_PER_READING = 8


def add_pronunciation(entries, key, syllables, smap, merged_erhua=False):
    """Add one reading of key; skip it if any syllable cannot be converted.

    Each syllable may have several IPA transcriptions (free variation); all
    combinations are emitted (capped), mirroring the upstream training
    dictionaries, so the aligner can pick the best one acoustically.

    Phones are grouped per character and written with " || " between groups.
    With merged_erhua the last syllable covers the last two characters; its
    phones are split as "all but the last" (the previous character) + "the
    last" (the rhotic tail of 儿).
    """
    variant_lists = []
    for syl in syllables:
        prons = smap.pronunciations(syl)
        if not prons:
            return False
        variant_lists.append(prons)

    bucket = entries.setdefault(key, [])
    added = 0
    for combo in itertools.product(*variant_lists):
        if added >= MAX_PRONUNCIATIONS_PER_READING:
            break
        if merged_erhua:
            last = list(combo[-1])
            if len(last) < 2:
                continue
            groups = [list(ps) for ps in combo[:-1]] + [last[:-1], last[-1:]]
        else:
            groups = [list(ps) for ps in combo]
        flat = tuple(p for g in groups for p in g)
        if all(tuple(p for g in e for p in g) != flat for e in bucket):
            bucket.append(groups)
            added += 1
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="",
        help="MFA acoustic model directory (contains meta.json + phones.txt). "
        "Default: auto-detect under the script directory.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output lexicon path. Default: lexicon.txt next to this script.",
    )
    args = parser.parse_args()

    # absolute() keeps symlink paths intact: the per-model
    # generate_lexicon.py files are symlinks to this script and the model
    # directory lives under the symlink's directory, not the real one.
    script_dir = Path(__file__).absolute().parent
    model_dir = Path(args.model_dir) if args.model_dir else find_model_dir(script_dir)
    out_path = Path(args.out) if args.out else script_dir / "lexicon.txt"

    inventory = load_inventory(model_dir)
    dict_path = find_pinyin_dictionary(script_dir, model_dir)
    table = load_pinyin_dictionary(dict_path) if dict_path else {}
    smap = SyllableMap(inventory, table)

    entries = {}  # key -> [phone tuple, ...]
    n_chars = n_phrases = 0

    # 1) single characters: every reading of a polyphone becomes its own line
    for cp in sorted(pinyin_dict.pinyin_dict):
        if not (0x3400 <= cp <= 0x9FFF):
            continue
        ch = chr(cp)
        n_chars += 1
        for syl in char_readings(ch):
            add_pronunciation(entries, ch, [syl], smap)

    # 2) phrases: contextual readings (polyphone readings pinned by context)
    for word in sorted(phrases_dict.phrases_dict):
        syllables = phrase_readings(word)
        n_phrases += 1
        add_pronunciation(entries, word, syllables, smap)
        merged = erhua_variant(word, syllables)
        if merged is not None:
            add_pronunciation(entries, word, merged, smap, merged_erhua=True)

    # 3) write: one pronunciation per line, no probability columns; " || "
    # separates the per-character phone groups
    n_lines = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for key in sorted(entries):
            for groups in entries[key]:
                if len(groups) == 1:
                    f.write(key + " " + " ".join(groups[0]) + "\n")
                else:
                    f.write(
                        key + " || " + " || ".join(" ".join(g) for g in groups)
                        + "\n"
                    )
                n_lines += 1

    n_phones = len(
        {p for prons in entries.values() for groups in prons for g in groups for p in g}
    )
    print(f"model dir:   {model_dir}")
    if dict_path:
        print(
            f"pinyin dict: {dict_path} ({len(table)} keys); "
            f"syllables: {smap.n_from_table} from dictionary, "
            f"{smap.n_from_conversion} via pinyin_to_ipa conversion"
        )
    else:
        print("pinyin dict: not found; all syllables via pinyin_to_ipa conversion")
    print(f"phone set:   {len(inventory)} phones (from phones.txt)")
    print(f"entries:     {n_chars} characters, {n_phrases} phrases")
    print(f"lexicon:     {n_lines} lines, {len(entries)} keys, {n_phones} phones used")
    if smap.failed:
        failed = " ".join(sorted(smap.failed))
        print(
            f"warning: {len(smap.failed)} pinyin syllables could not be "
            f"converted to the model's phones and were skipped: {failed}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
