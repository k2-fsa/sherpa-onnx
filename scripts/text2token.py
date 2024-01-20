#!/usr/bin/env python3

"""
This script encode the texts (given line by line through `text`) to tokens and
write the results to the file given by ``output``.

Usage:
If the tokens_type is bpe:

python3 ./text2token.py \
          --text texts.txt \
          --tokens tokens.txt \
          --tokens-type bpe \
          --bpe-model bpe.model \
          --output hotwords.txt

If the tokens_type is cjkchar:

python3 ./text2token.py \
          --text texts.txt \
          --tokens tokens.txt \
          --tokens-type cjkchar \
          --output hotwords.txt

If the tokens_type is cjkchar+bpe:

python3 ./text2token.py \
          --text texts.txt \
          --tokens tokens.txt \
          --tokens-type cjkchar+bpe \
          --bpe-model bpe.model \
          --output hotwords.txt

"""
import argparse

from sherpa_onnx import text2token


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="""Path to the input texts.

        Each line in the texts contains the original phrase, it might also contain some
        extra items, for example, the boosting score (startting with :), the triggering
        threshold (startting with #, only used in keyword spotting task) and the original
        phrase (startting with @). Note: extra items will be kept in the output.

        example input 1 (tokens_type = ppinyin):

        小爱同学 :2.0 #0.6 @小爱同学
        你好问问 :3.5 @你好问问
        小艺小艺 #0.6 @小艺小艺

        example output 1:

        x iǎo ài t óng x ué :2.0 #0.6 @小爱同学
        n ǐ h ǎo w èn w èn :3.5 @你好问问
        x iǎo y ì x iǎo y ì #0.6 @小艺小艺

        example input 2 (tokens_type = bpe):

        HELLO WORLD :1.5 #0.4
        HI GOOGLE :2.0 #0.8
        HEY SIRI #0.35

        example output 2:

        ▁HE LL O ▁WORLD :1.5 #0.4
        ▁HI ▁GO O G LE :2.0 #0.8
        ▁HE Y ▁S I RI #0.35
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="The path to tokens.txt.",
    )

    parser.add_argument(
        "--tokens-type",
        type=str,
        required=True,
        choices=["cjkchar", "bpe", "cjkchar+bpe", "fpinyin", "ppinyin"],
        help="""The type of modeling units, should be cjkchar, bpe, cjkchar+bpe, fpinyin or ppinyin.
        fpinyin means full pinyin, each cjkchar has a pinyin(with tone).
        ppinyin means partial pinyin, it splits pinyin into initial and final,
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="The path to bpe.model. Only required when tokens-type is bpe or cjkchar+bpe.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the encoded tokens will be written to.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    texts = []
    # extra information like boosting score (start with :), triggering threshold (start with #)
    # original keyword (start with @)
    extra_info = []
    with open(args.text, "r", encoding="utf8") as f:
        for line in f:
            extra = []
            text = []
            toks = line.strip().split()
            for tok in toks:
                if tok[0] == ":" or tok[0] == "#" or tok[0] == "@":
                    extra.append(tok)
                else:
                    text.append(tok)
            texts.append(" ".join(text))
            extra_info.append(extra)
    encoded_texts = text2token(
        texts,
        tokens=args.tokens,
        tokens_type=args.tokens_type,
        bpe_model=args.bpe_model,
    )
    with open(args.output, "w", encoding="utf8") as f:
        for i, txt in enumerate(encoded_texts):
            txt += extra_info[i]
            f.write(" ".join(txt) + "\n")


if __name__ == "__main__":
    main()
