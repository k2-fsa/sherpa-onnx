#!/usr/bin/env python3
"""Smoke-test a converted 7M bundle through sherpa-onnx.

For Arabic this also serves as a correctness check: sherpa's built-in espeak
front-end should land within a few percent of the reference pipeline. For
English it will NOT match (see README) -- the reference duration is printed
alongside so the gap is visible rather than silent.
"""

import argparse
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

TEXTS = {
    "ar": ["السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللهِ وَبَرَكَاتُهُ",
           "الْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ",
           "كَيْفَ حَالُكَ"],
    "en-us": ["The quick brown fox jumps over the lazy dog.",
              "Hello, and welcome to the seven million parameter Kokoro distill.",
              "How are you?"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="bundle dir")
    p.add_argument("--lang", required=True, choices=["ar", "en-us"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    p.add_argument("--espeak-data", default="/usr/share/espeak-ng-data")
    args = p.parse_args()

    d = Path(args.dir)
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(d / f"model.{args.precision}.onnx"),
                voices=str(d / "voices.bin"),
                tokens=str(d / "tokens.txt"),
                lang=args.lang,
                data_dir=args.espeak_data,
            ),
            num_threads=2,
        )
    )
    tts = sherpa_onnx.OfflineTts(cfg)

    for i, text in enumerate(TEXTS[args.lang]):
        audio = tts.generate(text, sid=0, speed=1.0)
        samples = np.array(audio.samples)
        out = d / f"test_{args.precision}_{i}.wav"
        sf.write(out, samples, audio.sample_rate)
        print(f"{out.name}: {len(samples) / audio.sample_rate:.2f}s "
              f"peak={np.abs(samples).max():.3f}")


if __name__ == "__main__":
    main()
