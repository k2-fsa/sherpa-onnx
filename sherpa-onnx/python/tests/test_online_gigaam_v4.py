# See scripts/gigaam-v4-streaming/README.md for the model and WAV.

import os
import unittest
from pathlib import Path

import sherpa_onnx
from test_online_recognizer import read_wave


@unittest.skipUnless(
    os.getenv("SHERPA_ONNX_GIGAAM_STREAMING_PACKAGE")
    and os.getenv("SHERPA_ONNX_GIGAAM_TEST_WAV"),
    "Set SHERPA_ONNX_GIGAAM_STREAMING_PACKAGE and SHERPA_ONNX_GIGAAM_TEST_WAV",
)
class TestOnlineGigaamV4(unittest.TestCase):
    def setUp(self):
        self.package = Path(os.environ["SHERPA_ONNX_GIGAAM_STREAMING_PACKAGE"])
        encoder = os.getenv(
            "SHERPA_ONNX_GIGAAM_STREAMING_ENCODER",
            "v4_rnnt_640ms_stream_step.int8.onnx",
        )
        self.config = dict(
            tokens=str(self.package / "tokens.e2e.txt"),
            encoder=str(self.package / encoder),
            decoder=str(self.package / "v4_rnnt_640ms_e2e_decoder.onnx"),
            joiner=str(self.package / "v4_rnnt_640ms_e2e_joint.onnx"),
            model_type="gigaam_v4",
            provider="cpu",
            num_threads=3,
            enable_endpoint_detection=False,
        )

    def test_partial_and_flush(self):
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**self.config)
        stream = recognizer.create_stream()
        samples, sample_rate = read_wave(os.environ["SHERPA_ONNX_GIGAAM_TEST_WAV"])
        for offset in range(0, len(samples), 10240):
            stream.accept_waveform(sample_rate, samples[offset : offset + 10240])
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

        partial = recognizer.tokens(stream)
        self.assertTrue(partial)
        stream.input_finished()
        self.assertTrue(recognizer.is_ready(stream))
        recognizer.decode_stream(stream)
        self.assertFalse(recognizer.is_ready(stream))
        self.assertEqual(recognizer.tokens(stream)[: len(partial)], partial)
        self.assertIn("родион", recognizer.get_result(stream).casefold())

    def test_endpoint_reset_preserves_queued_speech(self):
        config = {
            **self.config,
            "enable_endpoint_detection": True,
            "rule3_min_utterance_length": 0.5,
        }
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**config)
        samples, rate = read_wave(os.environ["SHERPA_ONNX_GIGAAM_TEST_WAV"])
        stream = recognizer.create_stream()
        stream.accept_waveform(rate, [0.0] * 10240)
        stream.accept_waveform(rate, samples[:5120])
        recognizer.decode_stream(stream)
        self.assertTrue(recognizer.is_endpoint(stream))
        self.assertEqual(recognizer.start_time(stream), 0.0)
        recognizer.reset(stream)
        self.assertAlmostEqual(recognizer.start_time(stream), 0.64)

        reference = recognizer.create_stream()
        reference.accept_waveform(rate, samples[:5120])
        for current in (stream, reference):
            current.accept_waveform(rate, samples[5120:10400])
            self.assertTrue(recognizer.is_ready(current))
            recognizer.decode_stream(current)
            current.accept_waveform(rate, samples[10400:])
            current.input_finished()
            while recognizer.is_ready(current):
                recognizer.decode_stream(current)

        expected = recognizer.tokens(reference)
        self.assertTrue(expected)
        self.assertEqual(recognizer.tokens(stream), expected)

    def test_rejects_mixed_heads(self):
        for arg, filename in (
            ("decoder", "v4_rnnt_640ms_raw_decoder.onnx"),
            ("joiner", "v4_rnnt_640ms_raw_joint.onnx"),
            ("tokens", "tokens.raw.txt"),
        ):
            with self.subTest(arg=arg):
                config = {**self.config, arg: str(self.package / filename)}
                with self.assertRaisesRegex(ValueError, "must use the same head"):
                    sherpa_onnx.OnlineRecognizer.from_transducer(**config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
