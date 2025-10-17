#!/usr/bin/env python3

from pypinyin import Style, lazy_pinyin, load_phrases_dict, phrases_dict, pinyin_dict

load_phrases_dict(
    {
        "行长": [["hang2"], ["zhang3"]],
        "银行行长": [["yin2"], ["hang2"], ["hang2"], ["zhang3"]],
    }
)


def main():
    filename = "lexicon.txt"

    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    i = 0
    with open(filename, "w", encoding="utf-8") as f:
        for key in word_dict:
            if not (0x4E00 <= key <= 0x9FFF):
                continue

            w = chr(key)
            tokens = lazy_pinyin(w, style=Style.TONE3, tone_sandhi=True)[0]
            if tokens == "shei2":
                tokens = "shui2"

            if tokens[-1] not in ("1", "2", "3", "4", "5"):
                tokens += "1"

            f.write(f"{w} {tokens}\n")

        for key in phrases:
            tokens = lazy_pinyin(key, style=Style.TONE3, tone_sandhi=True)
            for i in range(len(tokens)):
                if tokens[i] == "shei2":
                    tokens[i] = "shui2"

                if tokens[i][-1] not in ("1", "2", "3", "4", "5"):
                    tokens[i] += "1"

            tokens = " ".join(tokens)

            f.write(f"{key} {tokens}\n")


if __name__ == "__main__":
    main()
