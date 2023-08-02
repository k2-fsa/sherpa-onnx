#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-onnx Python API
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
# to download pre-trained models

import argparse
import sys
from pathlib import Path

from typing import List, Tuple
import sentencepiece as spm

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_onnx


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="""Used only when --decoding-method is modified_beam_search.
        It specifies number of active paths to keep during decoding.
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="",
        help="""
        Path to bpe.model, it will be used to tokenize contexts biasing phrases.
        Used only when --decoding-method=modified_beam_search
        """,
    )

    parser.add_argument(
        "--modeling-unit",
        type=str,
        default="char",
        help="""
        The type of modeling unit, it will be used to tokenize contexts biasing phrases.
        Valid values are bpe, bpe+char, char.
        Note: the char here means characters in CJK languages.
        Used only when --decoding-method=modified_beam_search
        """,
    )

    parser.add_argument(
        "--contexts",
        type=str,
        default="",
        help="""
        The context list, it is a string containing some words/phrases separated
        with /, for example, 'HELLO WORLD/I LOVE YOU/GO AWAY".
        Used only when --decoding-method=modified_beam_search
        """,
    )

    parser.add_argument(
        "--context-score",
        type=float,
        default=1.5,
        help="""
        The context score of each token for biasing word/phrase. Used only if
        --contexts is given.
        Used only when --decoding-method=modified_beam_search
        """,
    )

    return parser.parse_args()


def create_recognizer():
    args = get_args()
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_onnx.OnlineRecognizer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        context_score=args.context_score,
    )
    return recognizer

def encode_contexts(args, contexts: List[str]) -> List[List[int]]:
    sp = None
    if "bpe" in args.modeling_unit:
        assert_file_exists(args.bpe_model)
        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe_model)
    tokens = {}
    with open(args.tokens, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            assert len(toks) == 2, len(toks)
            assert toks[0] not in tokens, f"Duplicate token: {toks} "
            tokens[toks[0]] = int(toks[1])
    return sherpa_onnx.encode_contexts(
        modeling_unit=args.modeling_unit,
        contexts=contexts,
        sp=sp,
        tokens_table=tokens,
    )

def main():
    args = get_args()

    contexts_list = []
    contexts = [x.strip().upper() for x in args.contexts.split("/") if x.strip()]
    if contexts:
        print(f"Contexts list: {contexts}")
        contexts_list = encode_contexts(args, contexts)

    recognizer = create_recognizer()
    print("Started! Please speak")

    # The model is using 16 kHz, we use 48 kHz here to demonstrate that
    # sherpa-onnx will do resampling inside.
    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    if contexts_list:
        stream = recognizer.create_stream(contexts_list=contexts_list)
    else:
        stream = recognizer.create_stream()
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            result = recognizer.get_result(stream)
            if last_result != result:
                last_result = result
                print("\r{}".format(result), end="", flush=True)

if __name__ == "__main__":
    devices = sd.query_devices()
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
