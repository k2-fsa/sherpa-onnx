#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import numpy as np


def load_tokens(filename):
    ans = dict()
    i = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


logits = np.fromfile("./logits.raw", dtype=np.float32).reshape((-1, 25055))
print(logits.shape)

idx = logits.argmax(axis=-1)
print("idx", idx)
print(len(idx))
prev = -1
ids = []
for i in idx:
    if i != prev:
        ids.append(i)
    prev = i
ids = [i for i in ids if i != 0]
print(ids)

tokens = load_tokens("./tokens.txt")
text = "".join([tokens[i] for i in ids])

text = text.replace("_", " ")
print(text)
