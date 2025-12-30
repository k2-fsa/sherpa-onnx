#!/usr/bin/env python3

token2id = dict()
with open("./vocab_tts.txt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        fields = line.strip().split()
        if len(fields) == 0:
            token2id[" "] = i + 1
        else:
            token2id[fields[0]] = i + 1

with open("./tokens.txt", "w", encoding="utf-8") as f:
    for t, i in token2id.items():
        f.write(f"{t} {i}\n")
