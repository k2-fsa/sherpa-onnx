#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/fix_words_txt.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Regenerate graph/words.txt from the decoding graph's embedded output
# symbol table, when the provided words.txt disagrees with it.
#
# Word ids in the lattice are the graph's OUTPUT LABELS, so the graph's
# symbol table is the authoritative id -> word mapping (the vosk runtime
# maps via `g_fst_->OutputSymbols()`, words.txt only as a fallback).
# Some vosk models ship a words.txt whose ids DISAGREE with their graph
# (vosk-model-uk-v3-lgraph: 483/484 sampled ids differ -- the file is
# re-sorted), which maps correct decodes to garbage text.  Downstream
# consumers that read words.txt (e.g. the sherpa-onnx C++ runtime) then
# print wrong words, so the shipped file is rewritten to match the graph.
#
#   python3 fix_words_txt.py --model-dir ./vosk-model-uk-v3-lgraph-onnx \
#       [--note listing/<name>.words-note.txt]
#
# Prints "regenerated (N words)" or "unchanged".  With --note, writes a
# one-line note when regenerated (and removes a stale note otherwise);
# the note is shown on the model's info page.

import argparse
from pathlib import Path

import kaldifst


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="note file for the info page; written only when words.txt "
        "was regenerated (a stale note is removed)",
    )
    return parser.parse_args()


NOTE = (
    "`graph/words.txt` was regenerated from the decoding graph's output "
    "symbol table: the words.txt shipped with the original model assigns "
    "different ids to words than the graph does, and decoding word ids "
    "are graph output labels.  The graph files themselves are unchanged."
)


def main():
    args = get_args()
    model_dir = Path(args.model_dir)
    words_p = model_dir / "graph" / "words.txt"
    note_p = Path(args.note) if args.note else None

    def done(status, write_note):
        print(status)
        if note_p is not None:
            if write_note:
                note_p.parent.mkdir(parents=True, exist_ok=True)
                note_p.write_text(NOTE + "\n")
            elif note_p.is_file():
                note_p.unlink()

    # The graph's own output symbols: HCLG models carry them on
    # HCLG.fst, lgraph models on Gr.fst (HCLr o Gr keeps Gr's output
    # symbols).  NOTE: the symbol table is OWNED by the FST and the
    # Python wrapper does not keep its FST alive -- hold `fst` here or
    # the table dangles when the reader is a temporary (segfault).
    fst = None
    syms = None
    for name in ("HCLG.fst", "Gr.fst"):
        p = words_p.parent / name
        if p.is_file():
            fst = kaldifst.StdFst.read(str(p))
            syms = fst.output_symbols
            break
    if syms is None:
        done("unchanged (the graph carries no embedded symbol table)", False)
        return

    want = {}
    for wid in range(syms.available_key()):
        word = syms.find(wid)
        if word:
            want[wid] = word
    del fst  # the table is no longer needed

    if not words_p.is_file():
        # nothing provided to fix: the mapping lives in the graph
        # itself and every consumer that matters reads it from there
        done("unchanged (no words.txt to fix)", False)
        return

    have = {}
    for line in words_p.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            have[int(parts[1])] = parts[0]

    if have == want:
        done("unchanged", False)
        return

    tmp = words_p.with_name(words_p.name + ".part")
    with open(tmp, "w") as f:
        for wid in sorted(want):
            f.write(f"{want[wid]} {wid}\n")
    tmp.replace(words_p)
    done(f"regenerated ({len(want)} words)", True)


if __name__ == "__main__":
    main()
