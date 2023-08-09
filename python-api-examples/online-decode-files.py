#!/usr/bin/env python3

"""
This file demonstrates how to use sherpa-onnx Python API to transcribe
file(s) with a streaming model.

Usage:
    ./online-decode-files.py \
      /path/to/foo.wav \
      /path/to/bar.wav \
      /path/to/16kHz.wav \
      /path/to/8kHz.wav

Please refer to
https://k2-fsa.github.io/sherpa/onnx/index.html
to install sherpa-onnx and to download the pre-trained models
used in this file.
"""
import argparse
import time
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sentencepiece as spm
import sherpa_onnx


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
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
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
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

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to decode. Each file must be of WAVE"
        "format with a single channel, and each sample has 16-bit, "
        "i.e., int16_t. "
        "The sample rate of the file can be arbitrary and does not need to "
        "be 16 kHz",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


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
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        provider=args.provider,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        context_score=args.context_score,
    )

    print("Started!")
    start_time = time.time()

    contexts_list = []
    contexts = [x.strip().upper() for x in args.contexts.split("/") if x.strip()]
    if contexts:
        print(f"Contexts list: {contexts}")
        contexts_list = encode_contexts(args, contexts)

    streams = []
    total_duration = 0
    for wave_filename in args.sound_files:
        assert_file_exists(wave_filename)
        samples, sample_rate = read_wave(wave_filename)
        duration = len(samples) / sample_rate
        total_duration += duration

        if contexts_list:
            s = recognizer.create_stream(contexts_list=contexts_list)
        else:
            s = recognizer.create_stream()

        s.accept_waveform(sample_rate, samples)

        tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, tail_paddings)

        s.input_finished()

        streams.append(s)

    while True:
        ready_list = []
        for s in streams:
            if recognizer.is_ready(s):
                ready_list.append(s)
        if len(ready_list) == 0:
            break
        recognizer.decode_streams(ready_list)
    results = [recognizer.get_result(s) for s in streams]
    end_time = time.time()
    print("Done!")

    for wave_filename, result in zip(args.sound_files, results):
        print(f"{wave_filename}\n{result}")
        print("-" * 10)

    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / total_duration
    print(f"num_threads: {args.num_threads}")
    print(f"decoding_method: {args.decoding_method}")
    print(f"Wave duration: {total_duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(
        f"Real time factor (RTF): {elapsed_seconds:.3f}/{total_duration:.3f} = {rtf:.3f}"
    )


if __name__ == "__main__":
    main()
