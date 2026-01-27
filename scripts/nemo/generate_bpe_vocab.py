#!/usr/bin/env python3
# Copyright (c)  2026  github.com/nullbio
#
# Generate bpe.vocab file from a NeMo model for use with hotwords in sherpa-onnx.
#
# The bpe.vocab file contains BPE tokens with their scores (merge priorities),
# which is required for hotword/keyword boosting with modified beam search.
#
# Usage:
#   # From a pretrained model name:
#   python generate_bpe_vocab.py --model nvidia/parakeet-tdt-0.6b-v2
#
#   # From a local .nemo file:
#   python generate_bpe_vocab.py --model ./parakeet-tdt-0.6b-v2.nemo
#
#   # Specify output path:
#   python generate_bpe_vocab.py --model nvidia/parakeet-tdt-0.6b-v2 --output ./bpe.vocab

import argparse
from pathlib import Path


def generate_bpe_vocab_from_tokenizer(sp, output_path: str):
    """
    Generate bpe.vocab file from a sentencepiece processor.

    Uses the original scores from the SentencePiece model, which represent
    BPE merge priorities. These scores ensure correct tokenization order
    when encoding hotwords.

    Args:
        sp: SentencePiece processor object (from tokenizer.tokenizer)
        output_path: Output path for bpe.vocab file
    """
    vocab_size = sp.get_piece_size()
    print(f"Vocabulary size: {vocab_size}")

    print(f"Writing bpe.vocab to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for token_id in range(vocab_size):
            token = sp.id_to_piece(token_id)
            score = sp.get_score(token_id)
            f.write(f"{token}\t{score}\n")

    print("Done!")
    return output_path


def generate_bpe_vocab_from_model(asr_model, output_path: str):
    """
    Generate bpe.vocab file from a loaded NeMo ASR model.

    Args:
        asr_model: Loaded NeMo ASR model object
        output_path: Output path for bpe.vocab file
    """
    sp = asr_model.tokenizer.tokenizer
    return generate_bpe_vocab_from_tokenizer(sp, output_path)


def generate_bpe_vocab(model_path: str, output_path: str):
    """
    Generate bpe.vocab file from a NeMo ASR model.

    Args:
        model_path: Path to .nemo file or HuggingFace model name (e.g., nvidia/parakeet-tdt-0.6b-v2)
        output_path: Output path for bpe.vocab file
    """
    import nemo.collections.asr as nemo_asr

    # Load model
    print(f"Loading model: {model_path}")
    if Path(model_path).is_file():
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path)

    return generate_bpe_vocab_from_model(asr_model, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate bpe.vocab file from a NeMo ASR model for hotword support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From HuggingFace model:
  python generate_bpe_vocab.py --model nvidia/parakeet-tdt-0.6b-v2

  # From local .nemo file:
  python generate_bpe_vocab.py --model ./my_model.nemo --output ./bpe.vocab
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="NeMo model name (e.g., nvidia/parakeet-tdt-0.6b-v2) or path to .nemo file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./bpe.vocab",
        help="Output path for bpe.vocab file (default: ./bpe.vocab)",
    )

    args = parser.parse_args()

    generate_bpe_vocab(
        model_path=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
