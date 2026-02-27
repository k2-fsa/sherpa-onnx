#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import base64
from test import BinTokenizer


def main():
    tokenizer = BinTokenizer("./tokenizer.bin")

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for idx, token_bytes in enumerate(tokenizer.tokens):
            b64 = base64.b64encode(token_bytes).decode("ascii")
            f.write(f"{b64} {idx}\n")

    print("Saved to ./tokens.txt")


if __name__ == "__main__":
    main()
