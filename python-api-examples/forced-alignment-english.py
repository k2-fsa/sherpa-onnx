#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation

"""Forced alignment of English speech with the sherpa-onnx Python API.

It aligns the test wav of the english-mfa-v3.1.0 model package with its
transcript and writes a Praat TextGrid. The model directory, its
pronunciation lexicon and the test wav are hard-coded below.

How to download the model (the package contains the model files, the
lexicon and the test wav used here):

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
  tar xf english-mfa-v3.1.0.tar.bz2

How to run:

  python3 ./python-api-examples/forced-alignment-english.py

Output:

  en.TextGrid in the current directory. Open it with Praat
  (https://www.fon.hum.uva.nl/praat/) to visualize the phone and word
  boundaries.

See also forced-alignment-chinese.py and textgrid-to-word-wavs.py.
"""

import time

import sherpa_onnx
import soundfile as sf


def main():
    model_dir = "./english-mfa-v3.1.0"
    wav = f"{model_dir}/en.wav"
    text = (
        "ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    audio, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    # audio is a 1-D float32 numpy array normalized to the range [-1, 1]
    # sample_rate does not need to be 16000 Hz

    config = sherpa_onnx.ForcedAlignerConfig()
    config.model_dir = model_dir
    config.lexicon = f"{model_dir}/lexicon.txt"

    begin_init = time.perf_counter()
    aligner = sherpa_onnx.ForcedAligner(config)
    s = aligner.create_stream()
    end_init = time.perf_counter()

    s.accept_waveform(sample_rate, audio)

    begin = time.perf_counter()
    result = aligner.compute(s, text)
    end = time.perf_counter()

    for tier in result.tiers:
        print(f"--- tier: {tier.name}")
        for interval in tier.intervals:
            print(f"{interval.start:.2f} - {interval.end:.2f}  {interval.text}")

    with open("en.TextGrid", "w", encoding="utf-8") as f:
        f.write(result.to_textgrid())
    print("wrote en.TextGrid")

    # time statistics, like sherpa-onnx-fa prints
    init_seconds = end_init - begin_init
    elapsed_seconds = end - begin
    audio_duration = len(audio) / sample_rate
    num_phones = len(result.tiers[0].intervals) if result.tiers else 0
    num_words = len(result.tiers[1].intervals) if len(result.tiers) > 1 else 0
    print(f"init is done in {init_seconds:.3f} s")
    print(f"Elapsed seconds: {elapsed_seconds:.3f} s")
    print(
        f"Real time factor (RTF): {elapsed_seconds:.3f} / {audio_duration:.3f}"
        f" = {elapsed_seconds / audio_duration:.3f}"
    )
    print(f"phones: {num_phones}, words: {num_words}")


if __name__ == "__main__":
    main()
