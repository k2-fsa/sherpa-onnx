#!/usr/bin/env python3

"""
This file demonstrates how to use sherpa-onnx Python API to transcribe
file(s) with a streaming model.

paraformer Usage:
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

transducer Usage:
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

Please refer to
https://k2-fsa.github.io/sherpa/onnx/index.html
to install sherpa-onnx and to download the pre-trained models
used in this file.
"""
import argparse
import time
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

from _sherpa_onnx import (
    OfflineFeatureExtractorConfig,
    OfflineRecognizer as _Recognizer,
    OfflineRecognizerConfig,
    OfflineStream,
    OfflineModelConfig,
    OfflineTransducerModelConfig,
    OfflineParaformerModelConfig,
)

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
        "--debug",
        type=bool,
        default=False,
        help="Valid values are greedy_search and modified_beam_search",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Number of threads for neural network computation",
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

def main():
    args = get_args()
    if len(args.encoder)>0:
        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)
    else:
        assert_file_exists(args.paraformer)
    assert_file_exists(args.tokens)

    assert args.num_threads > 0, args.num_threads

    model_config = OfflineModelConfig(
        transducer= OfflineTransducerModelConfig(
            encoder_filename="",
            decoder_filename="",
            joiner_filename=""
        ),
        paraformer=OfflineParaformerModelConfig(
            model=args.paraformer
        ),
        tokens=args.tokens,
        num_threads=args.num_threads,
        debug=args.debug
    )

    decoding_method = args.decoding_method

    feat_config = OfflineFeatureExtractorConfig(
        sampling_rate=args.sample_rate,
        feature_dim=args.feature_dim,
    )

    recognizer_config = OfflineRecognizerConfig(
        feat_config=feat_config,
        model_config=model_config,
        decoding_method=decoding_method,
    )

    print("config ok!")

    recognizer = _Recognizer(recognizer_config)

    print("Started!")
    start_time = time.time()

    streams = []
    total_duration = 0
    for wave_filename in args.sound_files:
        assert_file_exists(wave_filename)
        samples, sample_rate = read_wave(wave_filename)
        duration = len(samples) / sample_rate
        total_duration += duration

        s = recognizer.create_stream()
        s.accept_waveform(sample_rate, samples)

        tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, tail_paddings)

        streams.append(s)


    recognizer.decode_streams(streams)
    results = [s.result.text for s in streams]
    end_time = time.time()
    print("Done!")

    for wave_filename, result in zip(args.sound_files, results):
        print(f"{wave_filename}\n{result}")
        print("-" * 10)

    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / duration
    print(f"num_threads: {args.num_threads}")
    print(f"decoding_method: {args.decoding_method}")
    print(f"Wave duration: {duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(f"Real time factor (RTF): {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f}")


if __name__ == "__main__":
    main()
