# sherpa-onnx/python/tests/test_offline_tts_silence_scale.py
#
# Copyright (c)  2026  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_offline_tts_silence_scale_py

import unittest
from pathlib import Path

import sherpa_onnx

d = "/tmp/icefall-models"
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/tts/index.html
# to download pre-trained models for testing


def _create_kitten_tts(silence_scale: float):
    model_dir = f"{d}/kitten-micro-en-v0_8"
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kitten=sherpa_onnx.OfflineTtsKittenModelConfig(
                model=f"{model_dir}/model.onnx",
                voices=f"{model_dir}/voices.bin",
                tokens=f"{model_dir}/tokens.txt",
                lexicon=f"{d}/lexicon-en-us.txt",
            ),
            num_threads=1,
        ),
        silence_scale=silence_scale,
    )
    return sherpa_onnx.OfflineTts(config)


class TestOfflineTtsSilenceScale(unittest.TestCase):
    def test_generate_uses_config_silence_scale(self):
        model_dir = Path(f"{d}/kitten-micro-en-v0_8")
        if not model_dir.is_dir() or not Path(f"{d}/lexicon-en-us.txt").is_file():
            print("skipping test_generate_uses_config_silence_scale()")
            return

        # The pauses between these sentences are longer than 0.2 seconds,
        # so silence_scale changes the length of the generated audio.
        text = (
            "Hello there. This is a test of the pause between sentences. "
            "And here is the third sentence, with a comma."
        )

        gen_config = sherpa_onnx.GenerationConfig()
        gen_config.sid = 0
        gen_config.speed = 1.0
        gen_config.silence_scale = 1.0

        tts = _create_kitten_tts(silence_scale=1.0)
        expected = tts.generate(text, gen_config)
        audio = tts.generate(text, sid=0, speed=1.0)

        default_tts = _create_kitten_tts(silence_scale=0.2)
        shortened = default_tts.generate(text, sid=0, speed=1.0)

        # generate(text, sid, speed) must use OfflineTtsConfig.silence_scale
        self.assertEqual(len(audio.samples), len(expected.samples))

        # and the two configs must give different results
        self.assertLess(len(shortened.samples), len(audio.samples))


if __name__ == "__main__":
    unittest.main()
