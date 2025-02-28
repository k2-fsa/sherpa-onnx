#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import re
from typing import List, Tuple

from misaki import zh
from misaki.token import MToken
from misaki.zh_frontend import ZH_MAP
from pypinyin import load_phrases_dict, phrases_dict, pinyin_dict

user_dict = {
    "还田": [["huan2"], ["tian2"]],
    "行长": [["hang2"], ["zhang3"]],
    "银行行长": [["yin2"], ["hang2"], ["hang2"], ["zhang3"]],
}

load_phrases_dict(user_dict)

phrases_dict.phrases_dict.update(**user_dict)


def process_text(self, text, with_erhua=True):
    """
    This function is modified from
    https://github.com/hexgrad/misaki/blob/main/misaki/zh_frontend.py#L155

    Note that we have removed jieba.posseg.lcut().
    """
    seg_cut = [(text, "v")]
    seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)
    tokens = []
    seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)
    initials = []
    finals = []
    # pypinyin, g2pM
    for word, pos in seg_cut:
        if pos == "x" and "\u4E00" <= min(word) and max(word) <= "\u9FFF":
            pos = "X"
        elif pos != "x" and word in self.punc:
            pos = "x"
        tk = MToken(text=word, tag=pos, whitespace="")
        if pos in ("x", "eng"):
            if not word.isspace():
                if pos == "x" and word in self.punc:
                    tk.phonemes = word
                tokens.append(tk)
            elif tokens:
                tokens[-1].whitespace += word
            continue
        elif (
            tokens and tokens[-1].tag not in ("x", "eng") and not tokens[-1].whitespace
        ):
            tokens[-1].whitespace = "/"

        # g2p
        sub_initials, sub_finals = self._get_initials_finals(word)
        # tone sandhi
        sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
        # er hua
        if with_erhua:
            sub_initials, sub_finals = self._merge_erhua(
                sub_initials, sub_finals, word, pos
            )

        initials.append(sub_initials)
        finals.append(sub_finals)
        # assert len(sub_initials) == len(sub_finals) == len(word)

        # sum(iterable[, start])
        # initials = sum(initials, [])
        # finals = sum(finals, [])

        phones = []
        for c, v in zip(sub_initials, sub_finals):
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c:
                phones.append(c)
            # replace punctuation by ` `
            # if c and c in self.punc:
            #     phones.append(c)
            if v and (v not in self.punc or v != c):  # and v not in self.rhy_phns:
                phones.append(v)
        phones = "_".join(phones).replace("_eR", "_er").replace("R", "_R")
        phones = re.sub(r"(?=\d)", "_", phones).split("_")
        tk.phonemes = "".join(ZH_MAP.get(p, self.unk) for p in phones)
        tokens.append(tk)

    result = "".join(
        (self.unk if tk.phonemes is None else tk.phonemes) + tk.whitespace
        for tk in tokens
    )

    return result, tokens


def generate_chinese_lexicon():
    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    g2p = zh.ZHG2P(version="1.1")

    lexicon = []
    for key in word_dict:
        if not (0x4E00 <= key <= 0x9FFF):
            continue
        w = chr(key)
        tokens: str = process_text(g2p.frontend, w)[0]
        lexicon.append((w, tokens))

    for key in phrases:
        tokens: str = process_text(g2p.frontend, key)[0]
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
