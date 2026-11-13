# sherpa-onnx/python/tests/test_nemo_hotword_finalization.py
#
# Copyright (c)  2026 (authors: github.com/mitsu-h)

import base64
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sherpa_onnx

# These small ONNX fixtures use IR 9 and opset 13. They exercise the real
# recognizer and decoder without downloading models or requiring onnx at test
# time. The encoder ignores its input and returns frame indices [0, 1]. The
# decoder casts its last token and passes through two [1, 1, 1] state tensors.
# The joiner selects a row of the following table using the frame index:
#
#              A       B           X       <unk>   blank      duration 0 / 1
# frame 0:    10     -30         -30        -30     -30          -50 / 0
# frame 1:   -30   log(.00001)  log(.8)     -30   log(.19999)    -50 / 0
#
# A second joiner changes frame 1 to B=.25, X=.7, blank=.05 for a completed
# hotword control. With beam size 1 only duration 1 is selected. Therefore AX
# and A+blank first compete when they finish, isolating the final beam prune.
_ENCODER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb246iwIKHwoGZnJhbWVzEgtlbmNvZGVyX291"
    "dCIISWRlbnRpdHkKJgoGbGVuZ3RoEhJlbmNvZGVyX291dF9sZW5ndGgiCElkZW50aXR5Egdl"
    "bmNvZGVyKhoIAQgBCAIQAUIGZnJhbWVzSggAAAAAAACAPyoWCAEQB0IGbGVuZ3RoSggCAAAAAAAA"
    "AFofCghmZWF0dXJlcxITChEIARINCgIIAQoCCFAKAxIBVFodCg9mZWF0dXJlc19sZW5ndGgSCg"
    "oICAcSBAoCCAFiIQoLZW5jb2Rlcl9vdXQSEgoQCAESDAoCCAEKAggBCgIIAmIgChJlbmNvZGVy"
    "X291dF9sZW5ndGgSCgoICAcSBAoCCAFCBAoAEA1yDwoKdm9jYWJfc2l6ZRIBNHIXChJzdWJzYW"
    "1wbGluZ19mYWN0b3ISATFyFAoObm9ybWFsaXplX3R5cGUSAk5BchQKD3ByZWRfcm5uX2xheWVy"
    "cxIBMXIQCgtwcmVkX2hpZGRlbhIBMXIUCgN1cmwSDXN5bnRoZXRpYy10ZHRyDgoIZmVhdF9kaW"
    "0SAjgw"
)
_DECODER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb2465wMKKAoHdGFyZ2V0cxIMZmxvYXRfdGFy"
    "Z2V0IgRDYXN0KgkKAnRvGAGgAQIKLAoMZmxvYXRfdGFyZ2V0CgZzaGFwZTMSC2RlY29kZXJf"
    "b3V0IgdSZXNoYXBlCiYKDnRhcmdldHNfbGVuZ3RoEgpvdXRfbGVuZ3RoIghJZGVudGl0eQoeCgZz"
    "dGF0ZTASCm91dF9zdGF0ZTAiCElkZW50aXR5Ch4KBnN0YXRlMRIKb3V0X3N0YXRlMSIISWRlbnRp"
    "dHkSB2RlY29kZXIqJggDEAdCBnNoYXBlM0oYAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAWhkKB3Rhcmdl"
    "dHMSDgoMCAYSCAoCCAEKAggBWhwKDnRhcmdldHNfbGVuZ3RoEgoKCAgGEgQKAggBWhwKBnN0YXRl"
    "MBISChAIARIMCgIIAQoCCAEKAggBWhwKBnN0YXRlMRISChAIARIMCgIIAQoCCAEKAggBYiEKC2Rl"
    "Y29kZXJfb3V0EhIKEAgBEgwKAggBCgIIAQoCCAFiGAoKb3V0X2xlbmd0aBIKCggIBhIECgIIAWIg"
    "CgpvdXRfc3RhdGUwEhIKEAgBEgwKAggBCgIIAQoCCAFiIAoKb3V0X3N0YXRlMRISChAIARIMCgII"
    "AQoCCAEKAggBQgQKABAN"
)
_JOINER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb246wAIKJgoLZW5jb2Rlcl9vdXQSBmZyYW1l"
    "MyIEQ2FzdCoJCgJ0bxgHoAECCiAKBmZyYW1lMwoGc2hhcGUxEgVmcmFtZSIHUmVzaGFwZQorCgV0"
    "YWJsZQoFZnJhbWUSBmxvZ2l0cyIGR2F0aGVyKgsKBGF4aXMYAKABAhIGam9pbmVyKhYIARAHQgZz"
    "aGFwZTFKCAEAAAAAAAAAKkcIAggHEAFCBXRhYmxlSjgAACBBAADwwQAA8MEAAPDBAADwwQAASMIAAAAA"
    "AADwwfE0OMG+f2S+AADwwbMDzr8AAEjCAAAAAFohCgtlbmNvZGVyX291dBISChAIARIMCgIIAQoC"
    "CAEKAggBWiEKC2RlY29kZXJfb3V0EhIKEAgBEgwKAggBCgIIAQoCCAFiGAoGbG9naXRzEg4KDAgB"
    "EggKAggBCgIIB0IECgAQDQ=="
)
_MATCH_JOINER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb246wAIKJgoLZW5jb2Rlcl9vdXQSBmZyYW1l"
    "MyIEQ2FzdCoJCgJ0bxgHoAECCiAKBmZyYW1lMwoGc2hhcGUxEgVmcmFtZSIHUmVzaGFwZQorCgV0"
    "YWJsZQoFZnJhbWUSBmxvZ2l0cyIGR2F0aGVyKgsKBGF4aXMYAKABAhIGam9pbmVyKhYIARAHQgZz"
    "aGFwZTFKCAEAAAAAAAAAKkcIAggHEAFCBXRhYmxlSjgAACBBAADwwQAA8MEAAPDBAADwwQAASMIAAAAA"
    "AADwwRhysb8Znra+AADwwRS6P8AAAEjCAAAAAFohCgtlbmNvZGVyX291dBISChAIARIMCgIIAQoC"
    "CAEKAggBWiEKC2RlY29kZXJfb3V0EhIKEAgBEgwKAggBCgIIAQoCCAFiGAoGbG9naXRzEg4KDAgB"
    "EggKAggBCgIIB0IECgAQDQ=="
)


# Three-frame variants have tokens A/B/C/X/Y/<unk>/blank and hotword ABC.
# Frame 2 emits Y with logit 10 (all other tokens -30). The first variant
# retains the original frame-1 probabilities to isolate per-frame pruning.
# The candidate variant has B=.1, X=.7, blank=.2: B's prefix boost excludes X
# from the top-1 biased token candidates, so preserving an acoustic beam alone
# cannot recover the correct AXY. Both candidate and beam fallbacks are needed.
_MIDDLE_ENCODER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb246jwIKHwoGZnJhbWVzEgtlbmNvZGVyX291dCII"
    "SWRlbnRpdHkKJgoGbGVuZ3RoEhJlbmNvZGVyX291dF9sZW5ndGgiCElkZW50aXR5EgdlbmNvZGVy"
    "Kh4IAQgBCAMQAUIGZnJhbWVzSgwAAAAAAACAPwAAAEAqFggBEAdCBmxlbmd0aEoIAwAAAAAAAABa"
    "HwoIZmVhdHVyZXMSEwoRCAESDQoCCAEKAghQCgMSAVRaHQoPZmVhdHVyZXNfbGVuZ3RoEgoKCAgH"
    "EgQKAggBYiEKC2VuY29kZXJfb3V0EhIKEAgBEgwKAggBCgIIAQoCCANiIAoSZW5jb2Rlcl9vdXRf"
    "bGVuZ3RoEgoKCAgHEgQKAggBQgQKABANcg8KCnZvY2FiX3NpemUSATZyFwoSc3Vic2FtcGxpbmdf"
    "ZmFjdG9yEgExchQKDm5vcm1hbGl6ZV90eXBlEgJOQXIUCg9wcmVkX3Jubl9sYXllcnMSATFyEAoL"
    "cHJlZF9oaWRkZW4SATFyFAoDdXJsEg1zeW50aGV0aWMtdGR0cg4KCGZlYXRfZGltEgI4MA=="
)
_MIDDLE_JOINER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb2469AIKJgoLZW5jb2Rlcl9vdXQSBmZyYW1lMyIE"
    "Q2FzdCoJCgJ0bxgHoAECCiAKBmZyYW1lMwoGc2hhcGUxEgVmcmFtZSIHUmVzaGFwZQorCgV0YWJs"
    "ZQoFZnJhbWUSBmxvZ2l0cyIGR2F0aGVyKgsKBGF4aXMYAKABAhIGam9pbmVyKhYIARAHQgZzaGFw"
    "ZTFKCAEAAAAAAAAAKnsIAwgJEAFCBXRhYmxlSmwAACBBAADwwQAA8MEAAPDBAADwwQAA8MEAAPDB"
    "AABIwgAAAAAAAPDB8TQ4wQAA8MG+f2S+AADwwQAA8MGzA86/AABIwgAAAAAAAPDBAADwwQAA8MEA"
    "APDBAAAgQQAA8MEAAPDBAABIwgAAAABaIQoLZW5jb2Rlcl9vdXQSEgoQCAESDAoCCAEKAggBCgII"
    "AVohCgtkZWNvZGVyX291dBISChAIARIMCgIIAQoCCAEKAggBYhgKBmxvZ2l0cxIOCgwIARIICgII"
    "AQoCCAlCBAoAEA0="
)
_CANDIDATE_JOINER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb2469AIKJgoLZW5jb2Rlcl9vdXQSBmZyYW1lMyIE"
    "Q2FzdCoJCgJ0bxgHoAECCiAKBmZyYW1lMwoGc2hhcGUxEgVmcmFtZSIHUmVzaGFwZQorCgV0YWJs"
    "ZQoFZnJhbWUSBmxvZ2l0cyIGR2F0aGVyKgsKBGF4aXMYAKABAhIGam9pbmVyKhYIARAHQgZzaGFw"
    "ZTFKCAEAAAAAAAAAKnsIAwgJEAFCBXRhYmxlSmwAACBBAADwwQAA8MEAAPDBAADwwQAA8MEAAPDB"
    "AABIwgAAAAAAAPDBjl0TwAAA8MEZnra+AADwwQAA8MEQAs6/AABIwgAAAAAAAPDBAADwwQAA8MEA"
    "APDBAAAgQQAA8MEAAPDBAABIwgAAAABaIQoLZW5jb2Rlcl9vdXQSEgoQCAESDAoCCAEKAggBCgII"
    "AVohCgtkZWNvZGVyX291dBISChAIARIMCgIIAQoCCAEKAggBYhgKBmxvZ2l0cxIOCgwIARIICgII"
    "AQoCCAlCBAoAEA0="
)


# RNN-T has the same three frames, with no duration logits. Its joiner
# indexes a 21x7 table by frame*7+last_token. After emitting each frame's token
# it emits blank, ensuring only one normal symbol per frame. This exercises
# the fallback when token-duration candidates have no duration component.
_RNNT_ENCODER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb246jwIKHwoGZnJhbWVzEgtlbmNvZGVyX291dCII"
    "SWRlbnRpdHkKJgoGbGVuZ3RoEhJlbmNvZGVyX291dF9sZW5ndGgiCElkZW50aXR5EgdlbmNvZGVy"
    "Kh4IAQgBCAMQAUIGZnJhbWVzSgwAAAAAAACAPwAAAEAqFggBEAdCBmxlbmd0aEoIAwAAAAAAAABa"
    "HwoIZmVhdHVyZXMSEwoRCAESDQoCCAEKAghQCgMSAVRaHQoPZmVhdHVyZXNfbGVuZ3RoEgoKCAgH"
    "EgQKAggBYiEKC2VuY29kZXJfb3V0EhIKEAgBEgwKAggBCgIIAQoCCANiIAoSZW5jb2Rlcl9vdXRf"
    "bGVuZ3RoEgoKCAgHEgQKAggBQgQKABANcg8KCnZvY2FiX3NpemUSATZyFwoSc3Vic2FtcGxpbmdf"
    "ZmFjdG9yEgExchQKDm5vcm1hbGl6ZV90eXBlEgJOQXIUCg9wcmVkX3Jubl9sYXllcnMSATFyEAoL"
    "cHJlZF9oaWRkZW4SATFyFQoDdXJsEg5zeW50aGV0aWMtcm5udHIOCghmZWF0X2RpbRICODA="
)
_RNNT_JOINER = (
    "CAkSGXNoZXJwYS1ob3R3b3JkLXJlZ3Jlc3Npb2468QcKJgoLZW5jb2Rlcl9vdXQSBmZyYW1lMyIE"
    "Q2FzdCoJCgJ0bxgHoAECCiAKBmZyYW1lMwoGc2hhcGUxEgVmcmFtZSIHUmVzaGFwZQolCgtkZWNv"
    "ZGVyX291dBIFbGFzdDMiBENhc3QqCQoCdG8YB6ABAgoeCgVsYXN0MwoGc2hhcGUxEgRsYXN0IgdS"
    "ZXNoYXBlCh8KBWZyYW1lCgVzZXZlbhIKcm93X29mZnNldCIDTXVsChwKCnJvd19vZmZzZXQKBGxh"
    "c3QSA3JvdyIDQWRkCikKBXRhYmxlCgNyb3cSBmxvZ2l0cyIGR2F0aGVyKgsKBGF4aXMYAKABAhIG"
    "am9pbmVyKhYIARAHQgZzaGFwZTFKCAEAAAAAAAAAKhUIARAHQgVzZXZlbkoIBwAAAAAAAAAq3AQI"
    "FQgHEAFCBXRhYmxlSswEAADwwQAA8MEAAPDBAADwwQAA8MEAAPDBAAAgQQAA8MEAAPDBAADwwQAA"
    "8MEAAPDBAADwwQAAIEEAAPDBAADwwQAA8MEAAPDBAADwwQAA8MEAACBBAADwwQAA8MEAAPDBAADw"
    "wQAA8MEAAPDBAAAgQQAA8MEAAPDBAADwwQAA8MEAAPDBAADwwQAAIEEAAPDBAADwwQAA8MEAAPDB"
    "AADwwQAA8MEAACBBAAAgQQAA8MEAAPDBAADwwQAA8MEAAPDBAADwwQAA8MHxNDjBAADwwb5/ZL4A"
    "APDBAADwwbMDzr8AAPDBAADwwQAA8MEAAPDBAADwwQAA8MEAACBBAADwwQAA8MEAAPDBAADwwQAA"
    "8MEAAPDBAAAgQQAA8MEAAPDBAADwwQAA8MEAAPDBAADwwQAAIEEAAPDBAADwwQAA8MEAAPDBAADw"
    "wQAA8MEAACBBAADwwQAA8MEAAPDBAADwwQAA8MEAAPDBAAAgQQAA8MEAAPDBAADwwQAA8MEAAPDB"
    "AADwwQAAIEEAAPDBAADwwQAA8MEAAPDBAAAgQQAA8MEAAPDBAADwwQAA8MEAAPDBAADwwQAAIEEA"
    "APDBAADwwQAA8MEAAPDBAADwwQAA8MEAAPDBAADwwQAAIEEAAPDBAADwwQAA8MEAAPDBAAAgQQAA"
    "8MEAAPDBAADwwQAA8MEAAPDBAADwwQAA8MEAAPDBAAAgQQAA8MEAAPDBAADwwQAA8MEAAPDBAADw"
    "wQAAIEEAAPDBAADwwQAA8MEAAPDBAADwwQAA8MEAACBBWiEKC2VuY29kZXJfb3V0EhIKEAgBEgwK"
    "AggBCgIIAQoCCAFaIQoLZGVjb2Rlcl9vdXQSEgoQCAESDAoCCAEKAggBCgIIAWIYCgZsb2dpdHMS"
    "DgoMCAESCAoCCAEKAggHQgQKABAN"
)


class TestNeMoHotwordFinalization(unittest.TestCase):
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
            "A 0\nB 1\nX 2\n<unk> 3\n<blk> 4\n", encoding="utf-8"
        )
        (cls.directory / "hotwords.txt").write_text("A B\n", encoding="utf-8")
        middle = cls.directory / "middle"
        middle.mkdir()
        for name, encoded in (
            ("encoder", _MIDDLE_ENCODER),
            ("decoder", _DECODER),
            ("joiner", _MIDDLE_JOINER),
            ("candidate-joiner", _CANDIDATE_JOINER),
        ):
            (middle / (name + ".onnx")).write_bytes(base64.b64decode(encoded))
        (middle / "tokens.txt").write_text(
            "A 0\nB 1\nC 2\nX 3\nY 4\n<unk> 5\n<blk> 6\n", encoding="utf-8"
        )
        (middle / "hotwords.txt").write_text("A B C\n", encoding="utf-8")
        rnnt = cls.directory / "rnnt"
        rnnt.mkdir()
        for name, encoded in (
            ("encoder", _RNNT_ENCODER),
            ("decoder", _DECODER),
            ("joiner", _RNNT_JOINER),
        ):
            (rnnt / (name + ".onnx")).write_bytes(base64.b64decode(encoded))
        (rnnt / "tokens.txt").write_text(
            "A 0\nB 1\nC 2\nX 3\nY 4\n<unk> 5\n<blk> 6\n", encoding="utf-8"
        )
        (rnnt / "hotwords.txt").write_text("A B C\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def decode(
        self,
        score=None,
        completed_match=False,
        fallback_paths=None,
        intermediate=False,
        candidate_pruning=False,
        non_tdt=False,
    ):
        directory = self.directory / "middle" if intermediate else self.directory
        if non_tdt:
            directory = self.directory / "rnnt"
        joiner = "joiner.onnx"
        if completed_match:
            joiner = "match-joiner.onnx"
        if candidate_pruning:
            joiner = "candidate-joiner.onnx"
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(directory / "encoder.onnx"),
            decoder=str(directory / "decoder.onnx"),
            joiner=str(directory / joiner),
            tokens=str(directory / "tokens.txt"),
            model_type="nemo_transducer",
            decoding_method="modified_beam_search",
            max_active_paths=1,
            hotwords_file=str(directory / "hotwords.txt") if score is not None else "",
            hotwords_score=score if score is not None else 1.5,
        )
        stream = recognizer.create_stream()
        if fallback_paths is not None:
            stream.set_option("hotword_fallback_paths", str(fallback_paths))
        stream.accept_waveform(16000, np.zeros(16000, dtype=np.float32))
        recognizer.decode_stream(stream)
        return stream.result

    def test_unfinished_hotword_does_not_delete_suffix(self):
        baseline = self.decode()
        self.assertEqual(baseline.text, "AX")
        for score in (1.0, 2.0, 3.0, 4.0, 6.0):
            with self.subTest(score=score):
                result = self.decode(score=score, fallback_paths=1)
                self.assertEqual(result.text, "AX")
                self.assertEqual(result.tokens, baseline.tokens)
                np.testing.assert_allclose(
                    result.ys_log_probs, baseline.ys_log_probs, atol=1e-6
                )

    def test_fallback_preserves_suffix_before_final_frame(self):
        baseline = self.decode(intermediate=True)
        self.assertEqual(baseline.text, "AXY")
        for score in (2.0, 3.0, 4.0, 6.0):
            with self.subTest(score=score):
                default = self.decode(score=score, intermediate=True)
                disabled = self.decode(score=score, intermediate=True, fallback_paths=0)
                self.assertEqual(default.text, disabled.text)
                result = self.decode(score=score, intermediate=True, fallback_paths=1)
                self.assertEqual(result.text, baseline.text)
                np.testing.assert_allclose(
                    result.ys_log_probs, baseline.ys_log_probs, atol=1e-6
                )

    def test_fallback_preserves_acoustic_token_candidates(self):
        baseline = self.decode(intermediate=True, candidate_pruning=True)
        self.assertEqual(baseline.text, "AXY")
        result = self.decode(
            score=3.0, intermediate=True, candidate_pruning=True, fallback_paths=1
        )
        self.assertEqual(result.text, baseline.text)
        np.testing.assert_allclose(
            result.ys_log_probs, baseline.ys_log_probs, atol=1e-6
        )

    def test_fallback_without_hotwords_preserves_baseline(self):
        baseline = self.decode(intermediate=True)
        result = self.decode(intermediate=True, fallback_paths=1)
        self.assertEqual(result.text, baseline.text)
        np.testing.assert_allclose(
            result.ys_log_probs, baseline.ys_log_probs, atol=1e-6
        )

    def test_invalid_fallback_paths_use_zero(self):
        baseline = self.decode(score=3.0, intermediate=True, fallback_paths=0)
        for fallback_paths in (-1, 2):
            with self.subTest(fallback_paths=fallback_paths):
                result = self.decode(
                    score=3.0,
                    intermediate=True,
                    fallback_paths=fallback_paths,
                )
                self.assertEqual(result.text, baseline.text)
                np.testing.assert_allclose(
                    result.ys_log_probs, baseline.ys_log_probs, atol=1e-6
                )

    def test_fallback_preserves_rnnt_suffix(self):
        baseline = self.decode(non_tdt=True)
        self.assertEqual(baseline.text, "AXY")
        result = self.decode(score=3.0, non_tdt=True, fallback_paths=1)
        self.assertEqual(result.text, baseline.text)
        np.testing.assert_allclose(
            result.ys_log_probs, baseline.ys_log_probs, atol=1e-6
        )

    def test_completed_hotword_keeps_its_reward(self):
        baseline = self.decode(completed_match=True)
        self.assertEqual(baseline.text, "AX")
        for fallback_paths in (None, 0, 1):
            with self.subTest(fallback_paths=fallback_paths):
                result = self.decode(
                    score=3.0, completed_match=True, fallback_paths=fallback_paths
                )
                self.assertEqual(result.text, "AB")
                np.testing.assert_allclose(
                    result.ys_log_probs, [0.0, math.log(0.25)], atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
