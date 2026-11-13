#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation

"""Cut a wav into per-word clips according to a TextGrid's words tier.

This is a general utility: it takes the wav, the TextGrid and the output
directory from the command line.

It reads the words tier of a Praat TextGrid (the format written by
ForcedAlignmentResult.to_textgrid()) and writes one wav file per word
interval, named wNN.wav (NN = the 1-based position in time), plus a
manifest.json with the word label, the start/end times and the file name
of every clip. For Chinese the words tier has one interval per character,
so the clips are per character. This is how to extract the segment of
each word of an alignment.

Usage:
  python3 ./python-api-examples/textgrid-to-word-wavs.py \
      --wav x.wav --textgrid x.TextGrid --out-dir <out-dir>

For example, after running forced-alignment-english.py (whose header
documents how to download the model package):

  python3 ./python-api-examples/textgrid-to-word-wavs.py \
      --wav english-mfa-v3.1.0/en.wav \
      --textgrid en.TextGrid \
      --out-dir word-clips-en

Output (in the --out-dir directory, word-clips-en/ in the example above):
  word-clips-en/w01.wav ... word-clips-en/wNN.wav   per-word audio
                                                  (same sample rate/format)
  word-clips-en/manifest.json               [{word, start, end, file}, ...]

  manifest.json looks like:

    [
      {"word": "ask", "start": 0.06, "end": 0.45, "file": "w01.wav"},
      {"word": "not", "start": 0.45, "end": 0.64, "file": "w02.wav"},
      ...
    ]
"""

import argparse
import json
from pathlib import Path


def read_words_tier(textgrid_path):
    """Parse the words tier of a TextGrid into (start, end, label) tuples.

    Handles the long-format TextGrid written by
    ForcedAlignmentResult.to_textgrid():
    ``item [2]: ... name = "words" ... intervals [k]: xmin/xmax/text``.

    Args:
      textgrid_path:
        Path of the TextGrid file.

    Returns:
      List of (start, end, word) in time order.
    """
    lines = Path(textgrid_path).read_text(encoding="utf-8").splitlines()

    # locate the tier named "words"
    tier_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("item ["):
            block = "\n".join(lines[i : i + 4])
            if 'name = "words"' in block:
                tier_start = i
                break
    if tier_start is None:
        raise ValueError(f"{textgrid_path}: no words tier found")

    intervals = []
    i = tier_start + 1
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("item ["):
            break  # next tier
        if line.startswith("intervals ["):
            xmin = float(lines[i + 1].split("=", 1)[1])
            xmax = float(lines[i + 2].split("=", 1)[1])
            text = lines[i + 3].split("=", 1)[1].strip().strip('"')
            intervals.append((xmin, xmax, text))
            i += 4
        else:
            i += 1
    return intervals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True,
                    help="input wav (only the first channel is used)")
    ap.add_argument("--textgrid", required=True,
                    help="TextGrid written by the forced alignment")
    ap.add_argument("--out-dir", required=True,
                    help="output directory for the per-word clips")
    args = ap.parse_args()

    import soundfile as sf

    words = read_words_tier(args.textgrid)
    samples, sample_rate = sf.read(args.wav, dtype="int16", always_2d=True)
    samples = samples[:, 0]  # only use the first channel

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, (start, end, word) in enumerate(words, 1):
        i0 = int(round(start * sample_rate))
        i1 = int(round(end * sample_rate))
        clip = samples[i0:i1]
        name = f"w{k:02d}.wav"
        sf.write(out_dir / name, clip, sample_rate)
        manifest.append(
            {"word": word, "start": round(start, 4),
             "end": round(end, 4), "file": name}
        )
        print(f"{name}  {start:.2f}-{end:.2f}s  {word}")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest)} clips + manifest.json to {out_dir}/")


if __name__ == "__main__":
    main()
