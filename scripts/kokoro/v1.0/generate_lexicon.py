#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import json
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


def generate_english_lexicon(kind: str):
    assert kind in ("us", "gb"), kind
    # If you want to add new words, please add them to
    # the user_defined dict.
    user_defined = {
        "Kokoro": "kˈOkəɹO",
        "Misaki": "misˈɑki",
    }

    user_defined_lower = dict()
    for k, v in user_defined.items():
        user_defined_lower[k.lower()] = v

    with open(f"./{kind}_gold.json", encoding="utf-8") as f:
        gold = json.load(f)

    with open(f"./{kind}_silver.json", encoding="utf-8") as f:
        silver = json.load(f)

    # words in us_gold has a higher priority than those in s_silver, so
    # we put us_gold after us_silver below
    english = {**silver, **gold}

    lexicon = dict()
    for k, v in english.items():
        k_lower = k.lower()

        if k_lower in user_defined_lower:
            print(f"{k} already exist in the user defined dict. Skip adding")
            continue

        if isinstance(v, str):
            lexicon[k_lower] = v
        else:
            assert isinstance(v, dict), (k, v)
            assert "DEFAULT" in v, (k, v)
            lexicon[k_lower] = v["DEFAULT"]

    return list(user_defined_lower.items()) + list(lexicon.items())


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
    us = generate_english_lexicon("us")
    gb = generate_english_lexicon("gb")
    zh = generate_chinese_lexicon()

    save("lexicon-us-en.txt", us)
    save("lexicon-gb-en.txt", gb)
    save("lexicon-zh.txt", zh)


if __name__ == "__main__":
    main()
