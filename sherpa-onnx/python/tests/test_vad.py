# sherpa-onnx/python/tests/test_vad.py
#
# To run this single test, use
#
#  ctest --verbose -R  test_vad_py

import unittest
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

# Put silero_vad.onnx from
# https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
# and a 16 kHz, 16-bit mono speech.wav in this folder
d = "/tmp/sherpa-test-data/vad"


def read_wave(wave_filename: str) -> np.ndarray:
    with wave.open(wave_filename) as f:
        assert f.getframerate() == 16000, f.getframerate()
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return samples.astype(np.float32) / 32768


class TestVoiceActivityDetector(unittest.TestCase):
    def test_flush_in_the_middle_of_speech(self):
        model = f"{d}/silero_vad.onnx"
        wave_filename = f"{d}/speech.wav"

        if not Path(model).is_file() or not Path(wave_filename).is_file():
            print("skipping test_flush_in_the_middle_of_speech()")
            return

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = model
        config.sample_rate = 16000
        window_size = config.silero_vad.window_size

        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

        samples = read_wave(wave_filename)
        num_fed = 0
        while num_fed < len(samples) and not vad.is_speech_detected():
            chunk = samples[num_fed : num_fed + window_size]
            vad.accept_waveform(chunk)
            num_fed += len(chunk)
        self.assertTrue(vad.is_speech_detected())

        # fewer samples than a window, so they wait in the internal buffer
        chunk = samples[num_fed : num_fed + 300]
        vad.accept_waveform(chunk)
        num_fed += len(chunk)

        vad.flush()
        self.assertFalse(vad.empty())
        self.assertEqual(vad.front.start + len(vad.front.samples), num_fed)


if __name__ == "__main__":
    unittest.main()
