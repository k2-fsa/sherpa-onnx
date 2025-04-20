#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import json


def main():
    with open("Kokoro-82M/config.json") as f:
        config = json.load(f)
    vocab = config["vocab"]

    with open("tokens.txt", "w", encoding="utf-8") as f:
        for k, i in vocab.items():
            f.write(f"{k} {i}\n")


if __name__ == "__main__":
    main()
