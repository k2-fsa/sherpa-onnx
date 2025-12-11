#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


from pypinyin import Style, lazy_pinyin, load_phrases_dict, phrases_dict, pinyin_dict
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

load_phrases_dict(
    {
        "行长": [["hang2"], ["zhang3"]],
        "银行行长": [["yin2"], ["hang2"], ["hang2"], ["zhang3"]],
    }
)
user_defined = {
    "微调": ["wei1", "tiao2"],
    "这个": ["zhe4", "ge4"],
    "方便地": ["fang1", "bian2", "de1"],
}


def get_initial_final(token):
    if isinstance(token, list):
        ans = ""
        sep = ""
        for t in token:
            ans += sep + get_initial_final(t)
            sep = " "
        return ans

    initial = to_initials(token, strict=False)

    final = to_finals_tone3(
        token,
        strict=False,
        neutral_tone_with_five=True,
    )

    ans = ""
    if initial:
        ans = initial + "0"

    if final:
        ans += f" {final}"

    return ans


def main():
    filename = "lexicon.txt"

    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    with open(filename, "w", encoding="utf-8") as f:
        for key in word_dict:
            if not (0x4E00 <= key <= 0x9FFF):
                continue

            w = chr(key)
            token = lazy_pinyin(
                w,
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )[0]

            initial_final = get_initial_final(token)

            f.write(f"{w} {initial_final}\n")

        for key, value in user_defined.items():
            initial_final = get_initial_final(value)
            f.write(f"{key} {initial_final}\n")

        for key in phrases:
            if key in user_defined:
                continue
            token = lazy_pinyin(
                key,
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )
            initial_final = get_initial_final(token)

            f.write(f"{key} {initial_final}\n")


if __name__ == "__main__":
    main()
