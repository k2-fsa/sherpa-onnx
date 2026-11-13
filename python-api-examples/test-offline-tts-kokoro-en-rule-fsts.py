#!/usr/bin/env python3
# Copyright (c) 2026 Hedy AI
"""Kokoro lexicon TTS with the optional English number/symbol rule pack.

Download kokoro-en-v0_19.tar.bz2 and lexicon-en-us.txt from sherpa's existing
GitHub tts-models release. Build/unpack the rules as described in
scripts/kokoro/english-tn/README.md. This example works with the existing API.
"""
import argparse
from pathlib import Path
import sherpa_onnx
import soundfile as sf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir', type=Path, required=True)
p.add_argument('--lexicon', required=True)
p.add_argument('--rules-dir', type=Path, required=True)
p.add_argument('--text', default='The budget is 1969 dollars.')
p.add_argument('--output', default='kokoro-english-rules.wav')
a = p.parse_args()
config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(a.model_dir / 'model.onnx'),
            voices=str(a.model_dir / 'voices.bin'),
            tokens=str(a.model_dir / 'tokens.txt'),
            lexicon=a.lexicon,
        ),
        num_threads=2,
        provider='cpu',
    ),
    rule_fsts=','.join(str(a.rules_dir / name) for name in
                       ['english_tn.fst', 'english_symbols.fst']),
)
tts = sherpa_onnx.OfflineTts(config)
audio = tts.generate(a.text, sherpa_onnx.GenerationConfig())
assert len(audio.samples) > 0, 'No audio generated'
sf.write(a.output, audio.samples, audio.sample_rate)
print(f'Saved {a.output}')
