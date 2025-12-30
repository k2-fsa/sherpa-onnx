#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import json


def main():
    with open("vocab.json") as f:
        token2id = json.load(f)

    with open("tokens.txt", "w", encoding="utf-8") as f:
        for s, i in token2id.items():
            if s == "|":
                s = " "
            if s == "[PAD]":
                s = "<blk>"

            f.write(f"{s} {i}\n")


if __name__ == "__main__":
    main()
