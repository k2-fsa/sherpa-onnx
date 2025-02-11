#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

from pathlib import Path

import tokenizers
from onnxruntime.quantization import QuantType, quantize_dynamic


def generate_tokens():
    if Path("./tokens.txt").is_file():
        return
    print("Generating tokens.txt")
    tokenizer = tokenizers.Tokenizer.from_file("./tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(vocab_size):
            s = tokenizer.id_to_token(i).strip()
            f.write(f"{s}\t{i}\n")


def main():
    generate_tokens()

    # Note(fangjun): Don't use int8 for the preprocessor since it has
    # a larger impact on the accuracy
    for f in ["uncached_decode", "cached_decode", "encode"]:
        if Path(f"{f}.int8.onnx").is_file():
            continue

        print("processing", f)
        quantize_dynamic(
            model_input=f"{f}.onnx",
            model_output=f"{f}.int8.onnx",
            weight_type=QuantType.QInt8,
        )


if __name__ == "__main__":
    main()
