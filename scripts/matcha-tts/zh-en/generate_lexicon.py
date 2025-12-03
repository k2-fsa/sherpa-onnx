#!/usr/bin/env python3

from pypinyin import Style, pinyin, load_phrases_dict, phrases_dict, pinyin_dict

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
            tokens = pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)[0][0]

            if tokens == "shei2":
                tokens = "shui2"

            if tokens[-1] not in ("1", "2", "3", "4", "5"):
                tokens += "1"

            f.write(f"{w} {tokens}\n")

        for key, value in user_defined.items():
            f.write(f"{key} {' '.join(value)}\n")

        for key in phrases:
            if key in user_defined:
                continue
            tokens = pinyin(key, style=Style.TONE3, neutral_tone_with_five=True)
            for i in range(len(tokens)):
                if tokens[i] == "shei2":
                    tokens[i] = "shui2"

                if tokens[i][-1] not in ("1", "2", "3", "4", "5"):
                    tokens[i] += "1"

            flatten = [t[0] for t in tokens]

            tokens = " ".join(flatten)

            f.write(f"{key} {tokens}\n")


if __name__ == "__main__":
    main()
