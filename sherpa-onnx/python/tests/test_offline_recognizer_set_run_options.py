# sherpa-onnx/python/tests/test_offline_recognizer_set_run_options.py
#
# Copyright (c)  2026  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_offline_recognizer_set_run_options_py

import unittest
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx

d = "/tmp/icefall-models"
# Reuses the same fixture set as test_offline_recognizer.py — the
# transducer model under /tmp/icefall-models/sherpa-onnx-zipformer-en-2023-04-01.
# When the model is absent the tests print and return (matching the
# style of the surrounding test files).


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768
        return samples_float32, f.getframerate()


def _build_transducer():
    encoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx"
    decoder = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx"
    joiner = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx"
    tokens = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/tokens.txt"
    wave0 = f"{d}/sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav"

    if not Path(encoder).is_file():
        return None, None

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=1,
        provider="cpu",
    )
    return recognizer, wave0


class TestOfflineRecognizerSetRunOptions(unittest.TestCase):
    def test_set_run_options_config_entry_no_crash(self):
        recognizer, wave0 = _build_transducer()
        if recognizer is None:
            print("skipping test_set_run_options_config_entry_no_crash()")
            return

        recognizer.set_run_options_config_entry(
            "memory.enable_memory_arena_shrinkage", "cpu:0"
        )

        s = recognizer.create_stream()
        samples, sample_rate = read_wave(wave0)
        s.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(s)
        print(s.result.text)

    def test_set_run_options_dict_no_crash(self):
        recognizer, wave0 = _build_transducer()
        if recognizer is None:
            print("skipping test_set_run_options_dict_no_crash()")
            return

        recognizer.set_run_options(
            {"memory.enable_memory_arena_shrinkage": "cpu:0"}
        )

        s = recognizer.create_stream()
        samples, sample_rate = read_wave(wave0)
        s.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(s)
        print(s.result.text)

    def test_set_run_options_invalid_key_no_crash(self):
        # Unknown RunOptions keys must not crash the process — ORT
        # rejects them at AddConfigEntry time, sherpa-onnx swallows that
        # via the same code path that handles real keys. The recognizer
        # must remain usable afterwards.
        recognizer, wave0 = _build_transducer()
        if recognizer is None:
            print("skipping test_set_run_options_invalid_key_no_crash()")
            return

        try:
            recognizer.set_run_options_config_entry(
                "this.key.does.not.exist", "whatever"
            )
        except Exception as e:
            # Either silently accepted or raises — both are acceptable;
            # what matters is the process stays alive and a subsequent
            # decode still works.
            print(f"set_run_options_config_entry raised (expected): {e!r}")

        s = recognizer.create_stream()
        samples, sample_rate = read_wave(wave0)
        s.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(s)
        print(s.result.text)


if __name__ == "__main__":
    unittest.main()
