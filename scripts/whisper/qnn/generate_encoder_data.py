#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import glob
from pathlib import Path

import numpy as np

from test_torch import compute_feat


def main():
    wav_files = glob.glob("*.wav")
    features_name = []
    for w in wav_files:
        f = compute_feat(w).numpy()
        print(w, f.shape)
        name = Path(w).stem

        s = f"encoder-input-{name}.raw"
        f.tofile(s)
        features_name.append(s)

    with open("encoder-input-list.txt", "w") as f:
        for line in features_name:
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()
