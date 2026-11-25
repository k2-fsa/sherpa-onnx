#!/usr/bin/env python3

"""
This file shows how to use a target-speaker VAD (NOMO-PVAD).

A regular VAD tells you whether *anybody* is talking. A target-speaker VAD
tells you whether the *enrolled* speaker is talking, so you need one short
enrollment clip of that speaker before you start.

To make the difference visible, this script accepts several enrollment clips
and runs the same audio through the detector once per enrolled speaker. The
detected segments should be different for different speakers.

Usage

(1) Download the two models

The enrollment model is the 3d-speaker ERes2NetV2 model, which is the model
NOMO-PVAD was trained with:

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx

NOMO-PVAD itself is exported with scripts/nomo-pvad/export-onnx.py:

  python3 ./scripts/nomo-pvad/export-onnx.py --output ./nomo-pvad.onnx

(2) Run it

  python3 ./target-speaker-vad.py \
      --embedding-model ./3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx \
      --pvad-model ./nomo-pvad.onnx \
      --enroll alice=./examples/enroll.wav \
      --enroll bob=./examples/other.wav \
      ./examples/test.wav

The first enrollment clip contains the speaker who talks first in test.wav,
while the second one contains the speaker who talks later. A possible output
is given below:

  --- alice ---
  timeline: ██████████████████......................
  [0.07, 2.54] s  (2.47 s)
  1 segment(s), 2.47 s of target speech out of 5.66 s

  --- bob ---
  timeline: .....................███████████████████
  [3.11, 5.66] s  (2.55 s)
  1 segment(s), 2.55 s of target speech out of 5.66 s

Note that only the enrolled speaker ends up in a segment.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sherpa_onnx
import soundfile as sf


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html "
        "to download it"
    )


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        help="""Path to the 3d-speaker ERes2NetV2 model. It is used only once
        per speaker, to compute the enrollment d-vector.""",
    )

    parser.add_argument(
        "--pvad-model",
        type=str,
        required=True,
        help="Path to the NOMO-PVAD model",
    )

    parser.add_argument(
        "--enroll",
        type=str,
        action="append",
        required=True,
        help="""An enrollment clip in the form name=/path/to/wave.wav.
        It can be given multiple times; the detector is then run once for
        each enrolled speaker.""",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="""Probability above which a 160 ms chunk counts as the target
        speaker talking""",
    )

    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=0.5,
        help="In seconds",
    )

    parser.add_argument(
        "--min-speech-duration",
        type=float,
        default=0.25,
        help="In seconds",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="How many samples are fed to the detector at a time",
    )

    parser.add_argument(
        "input",
        type=str,
        help="The wave file to analyze",
    )

    return parser.parse_args()


def parse_enrollments(values: List[str]) -> List[Tuple[str, str]]:
    ans = []
    for i, v in enumerate(values):
        if "=" in v:
            name, filename = v.split("=", maxsplit=1)
        else:
            name, filename = f"speaker-{i}", v
        assert_file_exists(filename)
        ans.append((name, filename))
    return ans


def to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return samples

    print(f"{sample_rate} Hz is not supported, resampling to 16000 Hz")
    import librosa

    return librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)


def compute_embedding(
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor, filename: str
) -> List[float]:
    samples, sample_rate = load_audio(filename)
    samples = to_16k(samples, sample_rate)

    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=16000, waveform=samples)
    stream.input_finished()

    assert extractor.is_ready(stream), (
        f"The enrollment clip {filename} is too short. "
        "Please use at least one second of speech."
    )
    return extractor.compute(stream)


def detect(
    config: sherpa_onnx.VadModelConfig, samples: np.ndarray, window_size: int
) -> List[Tuple[float, float]]:
    """Return a list of (start, end) in seconds."""
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

    for i in range(0, len(samples), window_size):
        vad.accept_waveform(samples[i : i + window_size])

    vad.flush()

    segments = []
    while not vad.empty():
        segment = vad.front
        start = segment.start / 16000
        segments.append((start, start + len(segment.samples) / 16000))
        vad.pop()

    return segments


def render_timeline(
    segments: List[Tuple[float, float]], duration: float, cols: int = 40
) -> str:
    """Draw which parts of the input contain the target speaker."""
    ans = ["."] * cols
    for start, end in segments:
        i0 = min(cols - 1, int(start / duration * cols))
        i1 = min(cols, max(i0 + 1, round(end / duration * cols)))
        for i in range(i0, i1):
            ans[i] = "█"
    return "".join(ans)


def main():
    args = get_args()

    assert_file_exists(args.embedding_model)
    assert_file_exists(args.pvad_model)
    assert_file_exists(args.input)

    enrollments = parse_enrollments(args.enroll)

    samples, sample_rate = load_audio(args.input)
    samples = to_16k(samples, sample_rate)
    duration = len(samples) / 16000

    print(f"Analyzing {args.input}: {duration:.2f} s")

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.embedding_model,
            num_threads=1,
            provider="cpu",
        )
    )
    dim = extractor.dim
    print(f"Speaker embedding dim: {dim}")

    for name, filename in enrollments:
        print(f"\n--- {name} ---")
        embedding = compute_embedding(extractor, filename)
        print(f"Enrolled from {filename} ({len(embedding)}-d d-vector)")

        config = sherpa_onnx.VadModelConfig()
        config.nomo_pvad = sherpa_onnx.NomoPvadModelConfig(
            model=args.pvad_model,
            speaker_embedding=embedding,
            threshold=args.threshold,
            min_silence_duration=args.min_silence_duration,
            min_speech_duration=args.min_speech_duration,
        )
        config.sample_rate = 16000
        config.num_threads = 1
        config.provider = "cpu"

        segments = detect(config, samples, args.window_size)

        print(f"timeline: {render_timeline(segments, duration)}")
        for start, end in segments:
            print(f"[{start:.2f}, {end:.2f}] s  ({end - start:.2f} s)")

        total = sum(end - start for start, end in segments)
        print(
            f"{len(segments)} segment(s), {total:.2f} s of target speech "
            f"out of {duration:.2f} s"
        )


if __name__ == "__main__":
    main()
