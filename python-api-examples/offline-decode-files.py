#!/usr/bin/env python3
#
# Copyright (c)  2023 by manyeyes

"""
This file demonstrates how to use sherpa-onnx Python API to transcribe
file(s) with a non-streaming model.

(1) For paraformer
    ./python-api-examples/offline-decode-files.py  \
      --tokens=/path/to/tokens.txt \
      --paraformer=/path/to/paraformer.onnx \
      --num-threads=2 \
      --decoding-method=greedy_search \
      --debug=false \
      --sample-rate=16000 \
      --feature-dim=80 \
      /path/to/0.wav \
      /path/to/1.wav

(2) For transducer models from icefall
    ./python-api-examples/offline-decode-files.py  \
      --tokens=/path/to/tokens.txt \
      --encoder=/path/to/encoder.onnx \
      --decoder=/path/to/decoder.onnx \
      --joiner=/path/to/joiner.onnx \
      --num-threads=2 \
      --decoding-method=greedy_search \
      --debug=false \
      --sample-rate=16000 \
      --feature-dim=80 \
      /path/to/0.wav \
      /path/to/1.wav

(3) For CTC models from NeMo

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
        "--bpe-model",
        type=str,
        default="",
        help="""
        Path to bpe.model,
        Used only when --decoding-method=modified_beam_search
        """,
    )

    parser.add_argument(
        "--modeling-unit",
        type=str,
        default="char",
        help="""
        The type of modeling unit.
        Valid values are bpe, bpe+char, char.
        Note: the char here means characters in CJK languages.
        """,
    )

    parser.add_argument(
        "--contexts",
        type=str,
        default="",
        help="""
        The context list, it is a string containing some words/phrases separated
        with /, for example, 'HELLO WORLD/I LOVE YOU/GO AWAY".
        """,
    )

    parser.add_argument(
        "--context-score",
        type=float,
        default=1.5,
        help="""
        The context score of each token for biasing word/phrase. Used only if
        --contexts is given.
        """,
    )

    parser.add_argument(
        "--encoder",
        default="",
        type=str,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        default="",
        type=str,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        default="",
        type=str,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--nemo-ctc",
        default="",
        type=str,
        help="Path to the model.onnx from NeMo CTC",
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
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="""Sample rate of the feature extractor. Must match the one
        expected  by the model. Note: The input sound files can have a
        different sample rate from this argument.""",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the one expected by the model",
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
        modeling_unit=args.modeling_unit, contexts=contexts, sp=sp, tokens_table=tokens
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


def main():
    args = get_args()
    assert_file_exists(args.tokens)
    assert args.num_threads > 0, args.num_threads

    contexts_list = []
    if args.encoder:
        assert len(args.paraformer) == 0, args.paraformer
        assert len(args.nemo_ctc) == 0, args.nemo_ctc

        contexts = [x.strip().upper() for x in args.contexts.split("/") if x.strip()]
        if contexts:
            print(f"Contexts list: {contexts}")
            contexts_list = encode_contexts(args, contexts)

        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            context_score=args.context_score,
            debug=args.debug,
        )
    elif args.paraformer:
        assert len(args.nemo_ctc) == 0, args.nemo_ctc
        assert_file_exists(args.paraformer)

        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.nemo_ctc:
        recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=args.nemo_ctc,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    else:
        print("Please specify at least one model")
        return

    print("Started!")
    start_time = time.time()

    streams = []
    total_duration = 0
    for wave_filename in args.sound_files:
        assert_file_exists(wave_filename)
        samples, sample_rate = read_wave(wave_filename)
        duration = len(samples) / sample_rate
        total_duration += duration
        if contexts_list:
            assert len(args.paraformer) == 0, args.paraformer
            assert len(args.nemo_ctc) == 0, args.nemo_ctc
            s = recognizer.create_stream(contexts_list=contexts_list)
        else:
            s = recognizer.create_stream()
        s.accept_waveform(sample_rate, samples)

        streams.append(s)

    recognizer.decode_streams(streams)
    results = [s.result.text for s in streams]
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
