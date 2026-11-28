# See python-api-examples/gigaam-offline-rnnt.md for the model and WAV.

import os
import unittest
from pathlib import Path

import sherpa_onnx
from test_offline_recognizer import read_wave


@unittest.skipUnless(
    os.getenv("SHERPA_ONNX_GIGAAM_OFFLINE_PACKAGE")
    and os.getenv("SHERPA_ONNX_GIGAAM_TEST_WAV"),
    "Set SHERPA_ONNX_GIGAAM_OFFLINE_PACKAGE and SHERPA_ONNX_GIGAAM_TEST_WAV",
)
class TestGigaamOffline(unittest.TestCase):
    def setUp(self):
        package = Path(os.environ["SHERPA_ONNX_GIGAAM_OFFLINE_PACKAGE"])
        self.config = dict(
            encoder=str(package / "v4_rnnt_encoder.int8.onnx"),
            advance=str(package / "v4_rnnt_advance.onnx"),
            tokenizer=str(package / "v4_rnnt_e2e_tokenizer.model"),
            num_threads=3,
        )

    def test_single_and_multiple_streams(self):
        recognizer = sherpa_onnx.OfflineRecognizer.from_gigaam(**self.config)
        samples, sample_rate = read_wave(os.environ["SHERPA_ONNX_GIGAAM_TEST_WAV"])
        single, *group = [recognizer.create_stream() for _ in range(3)]
        for stream in (single, *group):
            stream.accept_waveform(sample_rate, samples)

        recognizer.decode_stream(single)
        self.assertIn("родион", single.result.text.casefold())
        recognizer.decode_streams(group)
        for stream in group:
            self.assertEqual(stream.result.text, single.result.text)
            self.assertEqual(stream.result.tokens, single.result.tokens)

    def test_wrong_graph(self):
        self.config["encoder"] = self.config["advance"]
        with self.assertRaisesRegex(ValueError, "graph inputs/outputs"):
            sherpa_onnx.OfflineRecognizer.from_gigaam(**self.config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
