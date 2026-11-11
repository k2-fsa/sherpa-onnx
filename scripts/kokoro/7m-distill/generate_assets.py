#!/usr/bin/env python3
"""Generate tokens.txt and voices.bin for a 7M Kokoro distill.

tokens.txt must mirror the model's TRAINED vocabulary (config.json), not the
official 178-token Kokoro table. The two differ: this English distill assigns
'A' -> 24 where the official table uses 17, and the Arabic distill parks the
MSA pharyngeals ʕ and ħ on free embedding slots 7 and 8 which the official
table does not contain at those ids at all. Using the official file silently
mismaps embeddings.

voices.bin is the af_msa style pack flattened to float32 [510, 1, 256]. Both
students were conditioned on af_msa during distillation -- including the
English one, for which Kokoro's own af_heart degrades WER 0.0525 -> 0.0701.
"""

import argparse
import json
from pathlib import Path

import torch

# ʕ (ع) and ħ (ح) are absent from Kokoro's original vocabulary. The Arabic
# distill parks them on gap slots 7 and 8 so the ع/ء and ح/ه contrasts get
# their own embeddings instead of collapsing onto ʔ / h.
ARABIC_EXTRA_SYMBOLS = {"ʕ": 7, "ħ": 8}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["ar", "en"])
    p.add_argument("--dir", required=True, help="downloaded upstream model dir")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    src, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vocab = dict(json.loads((src / "config.json").read_text())["vocab"])
    if args.lang == "ar":
        for char, idx in ARABIC_EXTRA_SYMBOLS.items():
            clash = [c for c, i in vocab.items() if i == idx]
            if clash:
                raise SystemExit(f"slot {idx} for {char!r} already used by {clash}")
            vocab[char] = idx

    with (out / "tokens.txt").open("w") as f:
        for char, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
            f.write(f"{char} {idx}\n")
    print(f"tokens.txt: {len(vocab)} entries")

    pack = torch.load(src / "af_msa.pt", map_location="cpu", weights_only=True)
    if tuple(pack.shape) != (510, 1, 256):
        raise SystemExit(f"unexpected style pack shape {tuple(pack.shape)}")
    pack.numpy().tofile(out / "voices.bin")
    print(f"voices.bin: {pack.numel() * 4} bytes")


if __name__ == "__main__":
    main()
