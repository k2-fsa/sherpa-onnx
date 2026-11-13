# sherpa-onnx/python/tests/test_generic_hotword_fallback.py
#
# Copyright (c)  2026 (authors: github.com/mitsu-h)

import base64
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sherpa_onnx

# ONNX IR 9/opset 13 fixtures for the ordinary stateless transducer, with
# blank=0, unk=1, A=2, B=3, C=4, X=5, Y=6 and decoder context_size=1.
# The decoder casts its [N, 1] integer input to float. The encoder returns
# frame indices [0, 1, 2], expanded to the input batch size. Inputs with more
# than 60 feature frames have output length 3, and shorter inputs length 2.
# The joiner gathers the following logit table using each frame index:
#
#              blank       A       B           X       Y      other tokens
# frame 0:      -30       10     -30         -30      -30          -30
# frame 1:  log(.19999)   -30   log(.00001)  log(.8)   -30          -30
# frame 2:  log(.2)       -30     -30         -30    log(.8)        -30
#
# With hotword ABC and beam 2, score >= 3 allows A+blank to retain its
# unfinished prefix reward. Its extensions can displace AXY before that
# reward is refunded. Protecting candidates after refunding their parent's
# unfinished prefix preserves AXY. A second joiner changes frame 1 to
# blank=.2, B=.1, X=.7 for a completed AB control with beam 3.
# Embedding the small fixtures avoids downloads and an onnx test dependency.

_ENCODER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb2465wMKHwoPZmVhdHVyZXNfbGVuZ3RoEgViYXRj"
    "aCIFU2hhcGUKLQoFYmF0Y2gKBHRhaWwSCW91dF9zaGFwZSIGQ29uY2F0KgsKBGF4aXMYAKABAgoo"
    "CgZmcmFtZXMKCW91dF9zaGFwZRILZW5jb2Rlcl9vdXQiBkV4cGFuZAorCg9mZWF0dXJlc19sZW5n"
    "dGgKCXRocmVzaG9sZBIEbG9uZyIHR3JlYXRlcgotCgRsb25nCgV0aHJlZQoDdHdvEhJlbmNvZGVy"
    "X291dF9sZW5ndGgiBVdoZXJlEgdlbmNvZGVyKh4IAQgDCAEQAUIGZnJhbWVzSgwAAAAAAACAPwAA"
    "AEAqHAgCEAdCBHRhaWxKEAMAAAAAAAAAAQAAAAAAAAAqFxAHQgl0aHJlc2hvbGRKCDwAAAAAAAAA"
    "KhMQB0IFdGhyZWVKCAMAAAAAAAAAKhEQB0IDdHdvSggCAAAAAAAAAFogCghmZWF0dXJlcxIUChII"
    "ARIOCgMSAU4KAxIBVAoCCFBaHgoPZmVhdHVyZXNfbGVuZ3RoEgsKCQgHEgUKAxIBTmIiCgtlbmNv"
    "ZGVyX291dBITChEIARINCgMSAU4KAggDCgIIAWIhChJlbmNvZGVyX291dF9sZW5ndGgSCwoJCAcS"
    "BQoDEgFOQgQKABAN"
)

_DECODER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb246bgonCgd0YXJnZXRzEgtkZWNvZGVyX291dCIE"
    "Q2FzdCoJCgJ0bxgBoAECEgdkZWNvZGVyWhoKB3RhcmdldHMSDwoNCAcSCQoDEgFOCgIIAWIeCgtk"
    "ZWNvZGVyX291dBIPCg0IARIJCgMSAU4KAggBQgQKABANcg8KCnZvY2FiX3NpemUSATdyEQoMY29u"
    "dGV4dF9zaXplEgEx"
)

_JOINER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb2461wIKJgoLZW5jb2Rlcl9vdXQSBmZyYW1lcyIE"
    "Q2FzdCoJCgJ0bxgHoAECCiAKBmZyYW1lcwoGc2hhcGUxEgVmcmFtZSIHUmVzaGFwZQorCgV0YWJs"
    "ZQoFZnJhbWUSBmxvZ2l0cyIGR2F0aGVyKgsKBGF4aXMYAKABAhIGam9pbmVyKhYIARAHQgZzaGFw"
    "ZTFKCP//////////KmMIAwgHEAFCBXRhYmxlSlQAAPDBAADwwQAAIEEAAPDBAADwwQAA8MEAAPDB"
    "swPOvwAA8MEAAPDB8TQ4wQAA8MG+f2S+AADwwRACzr8AAPDBAADwwQAA8MEAAPDBAADwwb5/ZL5a"
    "HgoLZW5jb2Rlcl9vdXQSDwoNCAESCQoDEgFOCgIIAVoeCgtkZWNvZGVyX291dBIPCg0IARIJCgMS"
    "AU4KAggBYhkKBmxvZ2l0cxIPCg0IARIJCgMSAU4KAggHQgQKABAN"
)

_MATCH_JOINER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb2463QIKJgoLZW5jb2Rlcl9vdXQSBmZyYW1lcyIE"
    "Q2FzdCoJCgJ0bxgHoAECCiAKBmZyYW1lcwoGc2hhcGUxEgVmcmFtZSIHUmVzaGFwZQorCgV0YWJs"
    "ZQoFZnJhbWUSBmxvZ2l0cyIGR2F0aGVyKgsKBGF4aXMYAKABAhIMbWF0Y2gtam9pbmVyKhYIARAH"
    "QgZzaGFwZTFKCP//////////KmMIAwgHEAFCBXRhYmxlSlQAAPDBAADwwQAAIEEAAPDBAADwwQAA"
    "8MEAAPDBEALOvwAA8MEAAPDBjl0TwAAA8MEZnra+AADwwRACzr8AAPDBAADwwQAA8MEAAPDBAADw"
    "wb5/ZL5aHgoLZW5jb2Rlcl9vdXQSDwoNCAESCQoDEgFOCgIIAVoeCgtkZWNvZGVyX291dBIPCg0I"
    "ARIJCgMSAU4KAggBYhkKBmxvZ2l0cxIPCg0IARIJCgMSAU4KAggHQgQKABAN"
)


class TestGenericHotwordFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp_dir.name)
        for name, encoded in (
            ("encoder", _ENCODER),
            ("decoder", _DECODER),
            ("joiner", _JOINER),
            ("match-joiner", _MATCH_JOINER),
        ):
            (cls.directory / (name + ".onnx")).write_bytes(base64.b64decode(encoded))
        (cls.directory / "tokens.txt").write_text(
            "<blk> 0\n<unk> 1\nA 2\nB 3\nC 4\nX 5\nY 6\n", encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def recognizer(self, score=3.0, beam=2, completed_match=False):
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(self.directory / "encoder.onnx"),
            decoder=str(self.directory / "decoder.onnx"),
            joiner=str(
                self.directory
                / ("match-joiner.onnx" if completed_match else "joiner.onnx")
            ),
            tokens=str(self.directory / "tokens.txt"),
            model_type="transducer",
            decoding_method="modified_beam_search",
            max_active_paths=beam,
            hotwords_score=score,
        )

    def stream(self, recognizer, hotwords="A B C", fallback_paths=None, samples=16000):
        stream = recognizer.create_stream(hotwords=hotwords)
        if fallback_paths is not None:
            stream.set_option("hotword_fallback_paths", str(fallback_paths))
        stream.accept_waveform(16000, np.zeros(samples, dtype=np.float32))
        return stream

    def decode(self, score=3.0, fallback_paths=None, hotwords="A B C"):
        recognizer = self.recognizer(score=score)
        stream = self.stream(recognizer, hotwords, fallback_paths)
        recognizer.decode_stream(stream)
        return stream.result

    def test_high_score_suffix_deletion_and_fallback(self):
        baseline = self.decode(hotwords="")
        self.assertEqual(baseline.text, "AXY")
        for score in (1.0, 2.0):
            with self.subTest(low_score=score):
                self.assertEqual(self.decode(score=score).text, "AXY")
        # Keep the default search behavior explicit: fallback is opt-in.
        self.assertEqual(self.decode(score=3.0, fallback_paths=0).text, "AY")
        for score in (3.0, 4.0, 6.0):
            with self.subTest(high_score=score):
                result = self.decode(score=score, fallback_paths=2)
                self.assertEqual(result.text, "AXY")
                self.assertEqual(result.tokens, baseline.tokens)
                np.testing.assert_allclose(
                    result.ys_log_probs, baseline.ys_log_probs, atol=1e-6
                )

    def test_default_and_invalid_options_preserve_legacy_search(self):
        baseline = self.decode()
        for fallback_paths in (0, -1, 3):
            with self.subTest(fallback_paths=fallback_paths):
                result = self.decode(fallback_paths=fallback_paths)
                self.assertEqual(result.text, baseline.text)
                np.testing.assert_allclose(
                    result.ys_log_probs, baseline.ys_log_probs, atol=1e-6
                )

    def test_fallback_without_hotwords_preserves_baseline(self):
        baseline = self.decode(hotwords="")
        result = self.decode(hotwords="", fallback_paths=2)
        self.assertEqual(result.text, baseline.text)
        np.testing.assert_allclose(
            result.ys_log_probs, baseline.ys_log_probs, atol=1e-6
        )

    def test_completed_hotword_keeps_its_reward(self):
        recognizer = self.recognizer(beam=3, completed_match=True)
        baseline = self.stream(recognizer, hotwords="", samples=8000)
        recognizer.decode_stream(baseline)
        self.assertEqual(baseline.result.text, "AX")
        for fallback_paths in (None, 0, 1, 3):
            with self.subTest(fallback_paths=fallback_paths):
                stream = self.stream(recognizer, "A B", fallback_paths, samples=8000)
                recognizer.decode_stream(stream)
                self.assertEqual(stream.result.text, "AB")
                np.testing.assert_allclose(
                    stream.result.ys_log_probs, [0.0, math.log(0.1)], atol=1e-6
                )

    def test_batch_packing_keeps_per_stream_graphs_and_options(self):
        recognizer = self.recognizer()
        settings = (
            (8000, "A B C", 2, "AX"),
            (16000, "A B C", 0, "AY"),
            (16000, "", 2, "AXY"),
            (16000, "A B C", 2, "AXY"),
            (8000, "", 0, "AX"),
        )
        streams = [
            self.stream(recognizer, hotwords, fallback, samples)
            for samples, hotwords, fallback, _ in settings
        ]
        recognizer.decode_streams(streams)
        for i, (samples, hotwords, fallback, expected) in enumerate(settings):
            with self.subTest(stream=i):
                single = self.stream(recognizer, hotwords, fallback, samples)
                recognizer.decode_stream(single)
                self.assertEqual(streams[i].result.text, expected)
                self.assertEqual(streams[i].result.text, single.result.text)
                self.assertEqual(streams[i].result.tokens, single.result.tokens)
                self.assertEqual(streams[i].result.timestamps, single.result.timestamps)
                np.testing.assert_allclose(
                    streams[i].result.ys_log_probs,
                    single.result.ys_log_probs,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
