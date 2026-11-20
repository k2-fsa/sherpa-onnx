#!/usr/bin/env python3
# Copyright (c)  2026  graehl

"""
Save the first recordings of FLEURS's Galician test set (CC BY 4.0) as
0.wav, 1.wav, ... and their transcripts as transcript.txt.

Usage:

    ./download_test_wavs.py --num 4
"""

import argparse
import io
import urllib.request

import pyarrow.parquet as pq
import soundfile as sf

URL = "https://huggingface.co/api/datasets/google/fleurs/parquet/gl_es/test/0.parquet"


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num", type=int, default=4, help="How many waves")
    return parser.parse_args()


def main():
    args = get_args()
    with urllib.request.urlopen(URL) as f:
        table = pq.read_table(io.BytesIO(f.read()))
    rows = table.slice(0, args.num).to_pylist()
    with open("transcript.txt", "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            samples, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
            assert sample_rate == 16000, sample_rate
            sf.write(f"{i}.wav", samples, sample_rate, subtype="PCM_16")
            f.write(f"{i}.wav {row['transcription']}\n")
    print(f"saved {len(rows)} waves and transcript.txt")


if __name__ == "__main__":
    main()
