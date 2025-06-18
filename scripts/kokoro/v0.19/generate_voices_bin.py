#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import torch
from pathlib import Path


id2speaker = {
    0: "af",
    1: "af_bella",
    2: "af_nicole",
    3: "af_sarah",
    4: "af_sky",
    5: "am_adam",
    6: "am_michael",
    7: "bf_emma",
    8: "bf_isabella",
    9: "bm_george",
    10: "bm_lewis",
}

speaker2id = {speaker: idx for idx, speaker in id2speaker.items()}


def main():
    if Path("./voices.bin").is_file():
        print("./voices.bin exists - skip")
        return

    with open("voices.bin", "wb") as f:
        for _, speaker in id2speaker.items():
            m = torch.load(
                f"kLegacy/v0.19/voices/{speaker}.pt",
                weights_only=True,
                map_location="cpu",
            ).numpy()
            # m.shape (511, 1, 256)

            f.write(m.tobytes())


if __name__ == "__main__":
    main()
