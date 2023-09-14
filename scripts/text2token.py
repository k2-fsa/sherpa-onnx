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
        help="Path to the input texts",
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
        help="The type of modeling units, should be cjkchar, bpe or cjkchar+bpe",
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
    with open(args.text, "r", encoding="utf8") as f:
        for line in f:
            texts.append(line.strip())
    encoded_texts = text2token(
        texts,
        tokens=args.tokens,
        tokens_type=args.tokens_type,
        bpe_model=args.bpe_model,
    )
    with open(args.output, "w", encoding="utf8") as f:
        for txt in encoded_texts:
            f.write(" ".join(txt) + "\n")


if __name__ == "__main__":
    main()
