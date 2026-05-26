#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)

# It may take 100 minutes to generate the lexicon.
#
# You can download a pre-generated one from
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

from typing import List, Tuple

from pypinyin import phrases_dict, pinyin_dict

from piper.phonemize_chinese import ChinesePhonemizer


def generate_chinese_lexicon():
    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    phonemizer = ChinesePhonemizer(model_dir="./abc")

    lexicon = []
    for key in word_dict:
        if not (0x4E00 <= key <= 0x9FFF):
            continue
        w = chr(key)

        phonemes = phonemizer.phonemize(w)[0]
        tokens = []
        for p in phonemes:
            tokens.append(p)
            if p in {"1", "2", "3", "4", "5"}:
                tokens.append("_")

        lexicon.append((w, tokens))

    for key in phrases:
        phonemes = phonemizer.phonemize(key)[0]
        tokens = []
        for p in phonemes:
            tokens.append(p)
            if p in {"1", "2", "3", "4", "5"}:
                tokens.append("_")

        lexicon.append((key, tokens))

    return lexicon


def save(filename: str, lexicon: List[Tuple[str, List[str]]]):
    with open(filename, "w", encoding="utf-8") as f:
        for word, phones in lexicon:
            tokens = " ".join(phones)
            f.write(f"{word} {tokens}\n")


def main():
    zh = generate_chinese_lexicon()

    save("lexicon-zh-g2pw.txt", zh)


if __name__ == "__main__":
    main()
