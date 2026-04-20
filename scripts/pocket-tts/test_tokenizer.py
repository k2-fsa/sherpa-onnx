#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation

import json

import sentencepiece as spm


class SentencePieceBPETokenizer:
    def __init__(self, vocab_json, token_scores_json):
        with open(vocab_json, encoding="utf-8") as f:
            self.token2id = json.load(f)

        with open(token_scores_json, encoding="utf-8") as f:
            self.token2score = json.load(f)

        self.id2token = {v: k for k, v in self.token2id.items()}

        # index tokens by first char for speed
        self.by_first_char = {}
        for tok in self.token2id:
            if tok:
                self.by_first_char.setdefault(tok[0], []).append(tok)

        # byte fallback <0xNN>
        self.byte_token = {b: f"<0x{b:02X}>" for b in range(256)}

    def encode(self, text, return_type="ids"):
        text = text.replace(" ", "▁")
        if not text.startswith("▁"):
            text = "▁" + text

        n = len(text)
        dp = [-1e30] * (n + 1)
        back = [None] * (n + 1)
        dp[n] = 0.0

        for i in range(n - 1, -1, -1):
            c = text[i]

            for tok in self.by_first_char.get(c, []):
                if text.startswith(tok, i):
                    j = i + len(tok)
                    score = self.token2score[tok] + dp[j]
                    if score > dp[i]:
                        dp[i] = score
                        back[i] = tok

            # byte fallback
            if back[i] is None:
                b = text[i].encode("utf-8")[0]
                tok = self.byte_token[b]
                dp[i] = self.token2score[tok] + dp[i + 1]
                back[i] = tok

        # reconstruct
        tokens = []
        i = 0
        while i < n:
            tok = back[i]
            tokens.append(tok)
            i += len(tok)

        if return_type == "tokens":
            return tokens
        return [self.token2id[t] for t in tokens]


def main():
    tokenizer = SentencePieceBPETokenizer(
        vocab_json="./vocab.json", token_scores_json="./token_scores.json"
    )
    s = "Yesterday, I bought 3 apples, 2 bananas, and a dozen oranges. Wow! That's amazing—did you see it too? I can't believe it's already 10:30 p.m."

    tokens = tokenizer.encode(s, return_type="tokens")
    token_ids = tokenizer.encode(s, return_type="int")
    print(tokens)
    print(token_ids)
    sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
    #  print(help(sp.encode))

    gt_tokens = sp.encode(s, out_type=str)
    gt_token_ids = sp.encode(s, out_type=int)
    print(gt_tokens)
    print(len(tokens), len(gt_tokens))
    a = []
    for k, p in zip(tokens, gt_tokens):
        a.append(k == p)
    print(a)

    print(token_ids)
    print(gt_token_ids)


if __name__ == "__main__":
    main()
