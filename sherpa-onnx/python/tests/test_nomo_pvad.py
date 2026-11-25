# sherpa-onnx/python/tests/test_nomo_pvad.py
#
# To run this single test, use
#
#  ctest --verbose -R  test_nomo_pvad_py

import unittest
from pathlib import Path

import numpy as np
import sherpa_onnx

# Put nomo-pvad.onnx from
# https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/nomo-pvad.onnx
# in this folder. The model is 72 MB, so the tests that need it are skipped
# when it is absent.
d = "/tmp/sherpa-test-data/nomo-pvad"
model = f"{d}/nomo-pvad.onnx"


def skip_if_no_model():
    if not Path(model).is_file():
        print("skipping: %s does not exist" % model)
        return True
    return False


class TestNomoPvadModelConfig(unittest.TestCase):
    def test_default_values(self):
        config = sherpa_onnx.NomoPvadModelConfig()
        assert config.model == ""
        assert len(config.speaker_embedding) == 0
        assert config.speaker_embedding_file == ""
        assert abs(config.threshold - 0.7) < 1e-6
        assert abs(config.min_silence_duration - 0.5) < 1e-6
        assert abs(config.min_speech_duration - 0.25) < 1e-6
        assert abs(config.max_speech_duration - 20) < 1e-6
        assert config.neg_threshold < 0

    def test_str(self):
        config = sherpa_onnx.NomoPvadModelConfig(
            model="./nomo-pvad.onnx",
            speaker_embedding=[0.5] * 192,
            threshold=0.75,
        )
        s = str(config)
        assert "./nomo-pvad.onnx" in s, s
        assert "speaker_embedding_size=192" in s, s
        assert "threshold=0.75" in s, s

    def test_validate(self):
        config = sherpa_onnx.NomoPvadModelConfig()
        assert not config.validate()

        config.speaker_embedding = [0.5] * 192
        # The model file does not exist.
        config.model = "./no-such-file.onnx"
        assert not config.validate()

    def test_vad_model_config_has_nomo_pvad(self):
        config = sherpa_onnx.VadModelConfig()
        assert isinstance(config.nomo_pvad, sherpa_onnx.NomoPvadModelConfig)
        assert config.nomo_pvad.model == ""

        config.nomo_pvad = sherpa_onnx.NomoPvadModelConfig(
            model="./nomo-pvad.onnx", speaker_embedding=[0.5] * 192
        )
        assert config.nomo_pvad.model == "./nomo-pvad.onnx"
        assert "nomo_pvad" in str(config)


class TestNomoPvadModel(unittest.TestCase):
    def test_model_meta_data(self):
        if skip_if_no_model():
            return

        config = sherpa_onnx.VadModelConfig()
        config.nomo_pvad.model = model
        config.nomo_pvad.speaker_embedding = [1.0] * 192
        config.nomo_pvad.threshold = 0.7
        config.sample_rate = 16000

        v = sherpa_onnx.VadModel.create(config)
        assert v.window_size() == 2560, v.window_size()
        assert v.min_speech_duration_samples() == 4000
        assert v.min_silence_duration_samples() == 8000

    def test_digital_silence_has_no_speech(self):
        if skip_if_no_model():
            return

        config = sherpa_onnx.VadModelConfig()
        config.nomo_pvad.model = model
        config.nomo_pvad.speaker_embedding = [1.0] * 192
        config.nomo_pvad.threshold = 0.7
        config.sample_rate = 16000

        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

        samples = np.zeros(16000 * 3, dtype=np.float32)
        window = 512
        for i in range(0, len(samples), window):
            vad.accept_waveform(samples[i : i + window])

        vad.flush()
        assert vad.empty(), "silence must not produce any speech segment"


if __name__ == "__main__":
    unittest.main()
