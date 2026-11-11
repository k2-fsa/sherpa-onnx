#!/usr/bin/env python3

# This file shows how to use a non-streaming zipformer CTC model with an HLG
# decoding graph via sherpa-onnx's Python API.
#
# The HLG graph is passed to the C API through ctc_fst_decoder_config.
#
# clang-format off
#
# wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
# tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
# rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
#
# python3 ./python-api-examples/offline-zipformer-ctc-hlg-decode-files.py \
#   --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
#   --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx \
#   --graph ./sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst \
#   ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav
#
# clang-format on

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx


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
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model",
    )

    parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help="Path to H.fst, HL.fst, or HLG.fst",
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
        "--debug",
        type=int,
        default=0,
        help="Valid values: 1, 0",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="The input sound file to decode. It must be of WAVE"
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

    import wave

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
    print(vars(args))

    assert_file_exists(args.tokens)
    assert_file_exists(args.graph)
    assert_file_exists(args.model)

    # This is what the example exercises: the graph reaches the C API through
    # ctc_graph, which is passed to ctc_fst_decoder_config.
    recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
        model=args.model,
        tokens=args.tokens,
        num_threads=args.num_threads,
        provider=args.provider,
        debug=bool(args.debug),
        ctc_graph=args.graph,
    )

    audio, sample_rate = read_wave(args.sound_file)

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)

    result = stream.result
    print(args.sound_file)
    print(result.text)
    # When an HLG graph is used, the graph's output labels are in result.words.
    print(result.words)
    # str(result) is the JSON serialization of the result.
    print(result)


if __name__ == "__main__":
    main()
