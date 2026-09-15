#!/usr/bin/env python3
"""
Reference front-end for the 7M Kokoro distills (Arabic + English).

Why this file exists
--------------------
Both models were trained with a *specific* grapheme-to-phoneme front-end:

  * Nabra-7M-Distill (ar) : nabra_7m/arabic_g2p.py
        tashkeel -> misaki EspeakG2P("ar") -> clean_phonemes
        -> vocab from config.json, PLUS the two pharyngeals parked on the
           free Kokoro embedding slots 7 (ʕ) and 8 (ħ).
  * Kokoro-7M-Distill (en) : misaki.en.G2P (the stock Kokoro front-end)

sherpa-onnx's built-in Kokoro front-end calls espeak-ng directly and maps the
result through the *official 178-token* Kokoro table. That is NOT what either
7M model saw in training:

  - Arabic: espeak emits syllable-boundary '.', pharyngealization markers and
    the ʕ/ħ pharyngeals. Unprocessed, the dots become phantom pause tokens and
    ʕ/ħ fall outside the official table entirely, so they are dropped -> the
    ع/ء and ح/ه contrasts disappear and the prosody is wrong.
  - English: the official table assigns different ids than this distill's
    config.json (e.g. 'A' is 24 here, 17 officially) and misaki's phoneme
    choices differ from espeak's (dˈɔɡ vs dˈɑːɡ, ˈOvəɹ vs ˌo‍ʊvɚ). The 82M
    model is large enough to absorb that mismatch; a 7M distill is not.

So: run text through this module, feed the token ids straight to the ONNX
model. tokens.txt in each sherpa bundle mirrors the trained vocab exactly, so
a sherpa front-end that is fed these phonemes will agree with this code.

Usage
-----
    from g2p_frontend import ArabicFrontend, EnglishFrontend
    fe = ArabicFrontend()
    ids = fe("السَّلَامُ عَلَيْكُمْ")
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_vocab(config_path: Path) -> dict:
    return dict(json.loads(config_path.read_text())["vocab"])


class ArabicFrontend:
    """Text -> token ids for Nabra-7M-Distill, matching training exactly."""

    def __init__(self, model_dir: Path = REPO / "nabra_7m", diacritize: bool = True):
        sys.path.insert(0, str(model_dir))
        from arabic_g2p import ArabicG2P, EXTRA_SYMBOLS

        self.vocab = _load_vocab(model_dir / "config.json")
        # ʕ and ħ live on gap slots 7/8 -- they are absent from config.json's
        # vocab because Kokoro's original table never had them.
        for char, idx in EXTRA_SYMBOLS.items():
            clash = [c for c, i in self.vocab.items() if i == idx]
            if clash:
                raise RuntimeError(f"slot {idx} for {char!r} occupied by {clash}")
            self.vocab[char] = idx
        # Texts that already carry tashkeel are passed through untouched by
        # ArabicG2P.diacritize(), so diacritize=True is safe for both cases.
        self.g2p = ArabicG2P(diacritize=diacritize)

    def phonemes(self, text: str) -> str:
        return self.g2p.process(text)[1]

    def __call__(self, text: str) -> list[int]:
        ph = self.phonemes(text)
        unknown = sorted({c for c in ph if c not in self.vocab})
        if unknown:
            raise ValueError(f"phonemes outside trained vocab: {unknown}")
        return [self.vocab[c] for c in ph]


class EnglishFrontend:
    """Text -> token ids for Kokoro-7M-Distill (stock misaki front-end).

    NOTE on the voice pack: this student was distilled against `af_msa.pt`,
    not Kokoro's own `af_heart.pt`. Upstream measures WER 0.0525 -> 0.0701
    when af_heart is used instead, and it audibly skews the delivery.
    `af_heart.pt` ships only so older code keeps running.
    """

    VOICE = "af_msa.pt"

    def __init__(self, model_dir: Path = REPO / "kokoro_7m", british: bool = False):
        from misaki import en as misaki_en

        self.vocab = _load_vocab(model_dir / "config.json")
        self.g2p = misaki_en.G2P(trf=False, british=british, fallback=None)

    def phonemes(self, text: str) -> str:
        return self.g2p(text)[0]

    def __call__(self, text: str) -> list[int]:
        # misaki marks unresolvable words with '❓'; it is not a phoneme and is
        # absent from the vocab, so drop it rather than failing the sentence.
        return [self.vocab[c] for c in self.phonemes(text) if c in self.vocab]


def synthesize(onnx_path, voices_path, ids, speed=1.0):
    """Run token ids through an exported 7M ONNX model -> float32 waveform."""
    import numpy as np
    import onnxruntime as ort
    import torch

    voices = torch.load(voices_path, map_location="cpu", weights_only=True)
    # Kokoro indexes the style pack at len(phonemes) - 1 (see pipeline.py:251,
    # `pack[len(ps)-1]`). Using len(ps) picks the neighbouring style row and
    # audibly skews prosody, so keep the -1.
    ref_s = voices[len(ids) - 1].numpy().astype(np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tokens = np.array([[0] + list(ids) + [0]], dtype=np.int64)
    audio = sess.run(
        None,
        {
            "input_ids": tokens,
            "ref_s": ref_s,
            "speed": np.array([speed], dtype=np.float32),
        },
    )[0]
    return np.asarray(audio).reshape(-1)
