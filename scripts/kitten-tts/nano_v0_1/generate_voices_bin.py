#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
from pathlib import Path

import numpy as np

speakers = [
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
]

id2speaker = {idx: speaker for idx, speaker in enumerate(speakers)}

speaker2id = {speaker: idx for idx, speaker in id2speaker.items()}


def main():
    if Path("./voices.bin").is_file():
        print("./voices.bin exists - skip")
        return

    voices = np.load("./voices.npz")

    with open("voices.bin", "wb") as f:
        for speaker in speakers:
            v = voices[speaker]
            # v.shape (1, 256)
            f.write(v.tobytes())


if __name__ == "__main__":
    main()
