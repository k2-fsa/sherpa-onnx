#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import List, Tuple

from misaki import zh
from pypinyin import load_phrases_dict, phrases_dict, pinyin_dict

user_dict = {
    "还田": [["huan2"], ["tian2"]],
    "行长": [["hang2"], ["zhang3"]],
    "银行行长": [["yin2"], ["hang2"], ["hang2"], ["zhang3"]],
}

load_phrases_dict(user_dict)

phrases_dict.phrases_dict.update(**user_dict)


def generate_chinese_lexicon():
    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    g2p = zh.ZHG2P()
    lexicon = []

    for key in word_dict:
        if not (0x4E00 <= key <= 0x9FFF):
            continue
        w = chr(key)
        tokens: str = g2p.word2ipa(w)
        tokens = tokens.replace(chr(815), "")
        lexicon.append((w, tokens))

    for key in phrases:
        tokens: str = g2p.word2ipa(key)
        tokens = tokens.replace(chr(815), "")
        lexicon.append((key, tokens))
    return lexicon


def save(filename: str, lexicon: List[Tuple[str, str]]):
    with open(filename, "w", encoding="utf-8") as f:
        for word, phones in lexicon:
            tokens = " ".join(list(phones))
            f.write(f"{word} {tokens}\n")


def main():
    zh = generate_chinese_lexicon()

    save("lexicon-zh.txt", zh)


if __name__ == "__main__":
    main()
