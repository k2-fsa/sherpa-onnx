#!/usr/bin/env python3
# Copyright (c) 2026 Xiaomi Corporation

"""Model-free integration tests. Set SHERPA_ONNX_OFFLINE to a local build."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np


def save_model(path, nodes, inputs, outputs, initializers, metadata):
    import onnx
    from onnx import helper

    graph = helper.make_graph(nodes, path.stem, inputs, outputs, initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_models(directory):
    from onnx import TensorProto, helper, numpy_helper

    zeros = ",".join(["0"] * 80)
    ones = ",".join(["1"] * 80)
    metadata = {
        "mean": zeros,
        "invstd": ones,
        "model_type": "dolphin",
        "vocab_size": "10",
        "sos": "0",
        "eos": "1",
    }
    tensor = helper.make_tensor_value_info
    array = numpy_helper.from_array
    save_model(
        directory / "encoder.onnx",
        [helper.make_node("Identity", ["encoding"], ["encoder_out"])],
        [
            tensor("feats", TensorProto.FLOAT, [1, "T", 80]),
            tensor("feats_len", TensorProto.INT64, [1]),
        ],
        [tensor("encoder_out", TensorProto.FLOAT, [1, 8, 2])],
        [array(np.zeros((1, 8, 2), np.float32), "encoding")],
        metadata,
    )
    # The first text token depends on BOTH injected header fields.
    nodes = [
        helper.make_node("Shape", ["ys"], ["shape"]),
        helper.make_node("Gather", ["shape", "one"], ["length"], axis=0),
        helper.make_node("Gather", ["steps", "length"], ["default"], axis=0),
        helper.make_node("Equal", ["ys", "zh"], ["has_zh"]),
        helper.make_node("Equal", ["ys", "cn"], ["has_cn"]),
        helper.make_node("Cast", ["has_zh"], ["zh_int"], to=TensorProto.INT64),
        helper.make_node("Cast", ["has_cn"], ["cn_int"], to=TensorProto.INT64),
        helper.make_node("ReduceSum", ["zh_int"], ["zh_count"], keepdims=0),
        helper.make_node("ReduceSum", ["cn_int"], ["cn_count"], keepdims=0),
        helper.make_node("Mul", ["zh_count", "cn_count"], ["forced"]),
        helper.make_node("Add", ["eight", "forced"], ["text_token"]),
        helper.make_node("Equal", ["length", "five"], ["is_text"]),
        helper.make_node("Where", ["is_text", "text_token", "default"], ["token"]),
        helper.make_node("Unsqueeze", ["token", "axis"], ["token_vector"]),
        helper.make_node("OneHot", ["token_vector", "depth", "values"], ["logp"]),
    ]
    initializers = [
        array(np.array(value, np.int64), name)
        for name, value in [
            ("one", 1),
            ("five", 5),
            ("eight", 8),
            ("zh", 5),
            ("cn", 6),
            ("axis", [0]),
            ("depth", 10),
            ("steps", [0, 3, 4, 7, 2, 8, 1, 1, 1, 1]),
        ]
    ]
    initializers.append(array(np.array([-100, 0], np.float32), "values"))
    save_model(
        directory / "decoder.onnx",
        nodes,
        [
            tensor("encoder_out", TensorProto.FLOAT, [1, "T", 2]),
            tensor("ys", TensorProto.INT64, [1, "N"]),
        ],
        [tensor("logp", TensorProto.FLOAT, [1, 10])],
        initializers,
        metadata,
    )
    scores = np.full((1, 3, 10), -100, np.float32)
    scores[0, 0, 8] = 0
    scores[0, 1, 8] = 0
    scores[0, 2, 0] = 0
    save_model(
        directory / "ctc.onnx",
        [
            helper.make_node("Identity", ["scores"], ["log_probs"]),
            helper.make_node("Identity", ["length"], ["out_length"]),
        ],
        [
            tensor("feats", TensorProto.FLOAT, [1, "T", 80]),
            tensor("feats_len", TensorProto.INT64, [1]),
        ],
        [
            tensor("log_probs", TensorProto.FLOAT, [1, 3, 10]),
            tensor("out_length", TensorProto.INT64, [1]),
        ],
        [array(scores, "scores"), array(np.array([3], np.int64), "length")],
        metadata,
    )
    (directory / "tokens.txt").write_text(
        "<blk> 0\n<sos> 0\n<eos> 1\n<notimestamp> 2\n<en> 3\n<US> 4\n"
        "<zh> 5\n<CN> 6\n<asr> 7\nHELLO 8\nNIHAO 9\n",
        encoding="utf-8",
    )
    with wave.open(str(directory / "input.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(16000, np.int16).tobytes())


class DependencyStubsTest(unittest.TestCase):
    def test_missing_optional_dependencies(self):
        spec = importlib.util.spec_from_file_location(
            "dolphin_verifier", Path(__file__).with_name("test-onnx.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.dict("sys.modules"):
            import sys

            for name in list(sys.modules):
                if name.split(".")[0] in ("addict", "funasr", "modelscope"):
                    del sys.modules[name]
            with mock.patch("builtins.__import__", side_effect=ImportError):
                module.install_missing_dep_stubs()
            self.assertTrue(
                hasattr(
                    sys.modules["modelscope.models.audio.funasr.model"], "GenericFunASR"
                )
            )


@unittest.skipUnless(os.environ.get("SHERPA_ONNX_OFFLINE"), "Set SHERPA_ONNX_OFFLINE")
class NativeDolphinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="dolphin-test-")
        cls.directory = Path(cls.temp.name)
        make_models(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def decode(self, *options):
        command = [
            os.environ["SHERPA_ONNX_OFFLINE"],
            f"--dolphin-encoder={self.directory / 'encoder.onnx'}",
            f"--dolphin-decoder={self.directory / 'decoder.onnx'}",
            f"--tokens={self.directory / 'tokens.txt'}",
            *options,
            str(self.directory / "input.wav"),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_automatic_language(self):
        result = self.decode()
        self.assertEqual(result["lang"], "en")
        self.assertEqual(result["tokens"], ["HELLO"])

    @unittest.skipUnless(
        os.environ.get("SHERPA_ONNX_DOLPHIN_TEST"),
        "Set SHERPA_ONNX_DOLPHIN_TEST for native update tests",
    )
    def test_native_configuration_updates(self):
        env = dict(os.environ, SHERPA_ONNX_DOLPHIN_TEST_MODELS=str(self.directory))
        completed = subprocess.run(
            [env["SHERPA_ONNX_DOLPHIN_TEST"], "--gtest_filter=DolphinRecognizer.*"],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("[  PASSED  ] 1 test.", completed.stdout)

    def test_forced_language_and_region(self):
        result = self.decode("--dolphin-language=zh", "--dolphin-region=CN")
        self.assertEqual(result["lang"], "zh")
        self.assertEqual(result["tokens"], ["NIHAO"])

    def test_attention_takes_precedence_over_ctc(self):
        result = self.decode(
            f"--dolphin-model={self.directory / 'ctc.onnx'}",
            "--dolphin-language=zh",
            "--dolphin-region=CN",
        )
        self.assertEqual(result["tokens"], ["NIHAO"])

    def test_ctc_only_is_unchanged(self):
        completed = subprocess.run(
            [
                os.environ["SHERPA_ONNX_OFFLINE"],
                f"--dolphin-model={self.directory / 'ctc.onnx'}",
                f"--tokens={self.directory / 'tokens.txt'}",
                str(self.directory / "input.wav"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["tokens"], ["HELLO"])
        self.assertEqual(result["lang"], "")

    def test_forced_language_predicts_region(self):
        result = self.decode("--dolphin-language=zh")
        self.assertEqual(result["lang"], "zh")
        self.assertEqual(result["tokens"], ["HELLO"])

    def test_invalid_language_falls_back_without_exit(self):
        result = self.decode("--dolphin-language=unknown")
        self.assertEqual(result["lang"], "en")
        self.assertEqual(result["tokens"], ["HELLO"])


if __name__ == "__main__":
    unittest.main()
