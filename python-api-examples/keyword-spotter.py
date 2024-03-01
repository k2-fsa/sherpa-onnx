#!/usr/bin/env python3

"""
This file demonstrates how to use sherpa-onnx Python API to do keyword spotting
from wave file(s).

Please refer to
https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
to download pre-trained models.
"""
import argparse
import time
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
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
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="""
        It specifies number of active paths to keep during decoding.
        """,
    )

    parser.add_argument(
        "--num-trailing-blanks",
        type=int,
        default=1,
        help="""The number of trailing blanks a keyword should be followed. Setting
        to a larger value (e.g. 8) when your keywords has overlapping tokens
        between each other.
        """,
    )

    parser.add_argument(
        "--keywords-file",
        type=str,
        help="""
        The file containing keywords, one words/phrases per line, and for each
        phrase the bpe/cjkchar/pinyin are separated by a space. For example:

        ▁HE LL O ▁WORLD
        x iǎo ài t óng x ué 
        """,
    )

    parser.add_argument(
        "--keywords-score",
        type=float,
        default=1.0,
        help="""
        The boosting score of each token for keywords. The larger the easier to
        survive beam search.
        """,
    )

    parser.add_argument(
        "--keywords-threshold",
        type=float,
        default=0.25,
        help="""
        The trigger threshold (i.e. probability) of the keyword. The larger the
        harder to trigger.
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
        "https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html to download it"
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
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)

    assert Path(
        args.keywords_file
    ).is_file(), (
        f"keywords_file : {args.keywords_file} not exist, please provide a valid path."
    )

    keyword_spotter = sherpa_onnx.KeywordSpotter(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        max_active_paths=args.max_active_paths,
        keywords_file=args.keywords_file,
        keywords_score=args.keywords_score,
        keywords_threshold=args.keywords_threshold,
        num_trailing_blanks=args.num_trailing_blanks,
        provider=args.provider,
    )

    print("Started!")
    start_time = time.time()

    streams = []
    total_duration = 0
    for wave_filename in args.sound_files:
        assert_file_exists(wave_filename)
        samples, sample_rate = read_wave(wave_filename)
        duration = len(samples) / sample_rate
        total_duration += duration

        s = keyword_spotter.create_stream()

        s.accept_waveform(sample_rate, samples)

        tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, tail_paddings)

        s.input_finished()

        streams.append(s)

    results = [""] * len(streams)
    while True:
        ready_list = []
        for i, s in enumerate(streams):
            if keyword_spotter.is_ready(s):
                ready_list.append(s)
            r = keyword_spotter.get_result(s)
            if r:
                results[i] += f"{r}/"
                print(f"{r} is detected.")
        if len(ready_list) == 0:
            break
        keyword_spotter.decode_streams(ready_list)
    end_time = time.time()
    print("Done!")

    for wave_filename, result in zip(args.sound_files, results):
        print(f"{wave_filename}\n{result}")
        print("-" * 10)

    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / total_duration
    print(f"num_threads: {args.num_threads}")
    print(f"Wave duration: {total_duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(
        f"Real time factor (RTF): {elapsed_seconds:.3f}/{total_duration:.3f} = {rtf:.3f}"
    )


if __name__ == "__main__":
    main()
