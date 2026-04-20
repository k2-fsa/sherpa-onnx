#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation

import json

import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer.model")

token2id = {}
token2score = {}

for i in range(sp.get_piece_size()):
    tok = sp.id_to_piece(i)
    token2id[tok] = i
    token2score[tok] = sp.get_score(i)

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(token2id, f, indent=2, ensure_ascii=False)

with open("token_scores.json", "w", encoding="utf-8") as f:
    json.dump(token2score, f, indent=2, ensure_ascii=False)
