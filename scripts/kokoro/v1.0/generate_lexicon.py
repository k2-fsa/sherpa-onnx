#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import json
from pypinyin import phrases_dict, pinyin_dict
from misaki import zh


def generate_english_lexicon():
    # If you want to add new words, please add them to
    # the user_defined dict.
    user_defined = {
        "Kokoro": "kˈOkəɹO",
        "Misaki": "misˈɑki",
    }

    user_defined_lower = dict()
    for k, v in user_defined.items():
        user_defined_lower[k.lower()] = v

    with open("./us_gold.json", encoding="utf-8") as f:
        us_gold = json.load(f)

    with open("./us_silver.json", encoding="utf-8") as f:
        us_silver = json.load(f)

    # words in us_gold has a higher priority than those in s_silver, so
    # we put us_gold after us_silver below
    us = {**us_silver, **us_gold}

    lexicon = dict()
    for k, v in us.items():
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
        tokens: str = g2p(w)
        lexicon.append((w, tokens))

    for key in phrases:
        tokens: str = g2p(key)
        lexicon.append((key, tokens))
    return lexicon


def main():
    english = generate_english_lexicon()
    chinese = generate_chinese_lexicon()

    all_lang = english + chinese
    with open("lexicon.txt", "w", encoding="utf-8") as f:
        for word, phones in all_lang:
            tokens = " ".join(list(phones))
            f.write(f"{word} {tokens}\n")


if __name__ == "__main__":
    main()
