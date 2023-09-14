# Copyright (c)  2023  Xiaomi Corporation

import logging
import click
from pathlib import Path
from sherpa_onnx import text2token


@click.group()
def cli():
    """
    The shell entry point to sherpa-onnx.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )


@cli.command(name="text2token")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path())
@click.option(
    "--tokens",
    type=str,
    required=True,
    help="The path to tokens.txt.",
)
@click.option(
    "--tokens-type",
    type=str,
    required=True,
    help="The type of modeling units, should be cjkchar, bpe or cjkchar+bpe",
)
@click.option(
    "--bpe-model",
    type=str,
    help="The path to bpe.model. Only required when tokens-type is bpe or cjkchar+bpe.",
)
def encode_text(
    input: Path, output: Path, tokens: Path, tokens_type: str, bpe_model: Path
):
    """
    Encode the texts given by the INPUT to tokens and write the results to the OUTPUT.
    """
    texts = []
    with open(input, "r", encoding="utf8") as f:
        for line in f:
            texts.append(line.strip())
    encoded_texts = text2token(
        texts, tokens=tokens, tokens_type=tokens_type, bpe_model=bpe_model
    )
    with open(output, "w", encoding="utf8") as f:
        for txt in encoded_texts:
            f.write(" ".join(txt) + "\n")
