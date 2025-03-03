#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import torch
from pathlib import Path


speakers = [
    "af_maple",
    "af_sol",
    "bf_vale",
]
for i in range(1, 99 + 1):
    name = "zf_{:03d}".format(i)
    if Path(f"voices/{name}.pt").is_file():
        speakers.append(name)

for i in range(9, 100 + 1):
    name = "zm_{:03d}".format(i)
    if Path(f"voices/{name}.pt").is_file():
        speakers.append(name)


id2speaker = {index: value for index, value in enumerate(speakers)}

speaker2id = {speaker: idx for idx, speaker in id2speaker.items()}


def main():
    if Path("./voices.bin").is_file():
        print("./voices.bin exists - skip")
        return

    with open("voices.bin", "wb") as f:
        for _, speaker in id2speaker.items():
            m = torch.load(
                f"voices/{speaker}.pt",
                weights_only=True,
                map_location="cpu",
            ).numpy()
            # m.shape (510, 1, 256)

            f.write(m.tobytes())


if __name__ == "__main__":
    main()
