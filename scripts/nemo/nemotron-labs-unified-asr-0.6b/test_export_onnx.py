#!/usr/bin/env python3
# Copyright (c) 2026 Julian Pscheid

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

from export_onnx import merge_prompt


def model(nodes, inputs, outputs, initializers=()):
    return h.make_model(
        h.make_graph(nodes, "test", inputs, outputs, list(initializers)),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=8,
    )


class TestMergePrompt(unittest.TestCase):
    def test_output_order_language_input_and_external_weights(self):
        with tempfile.TemporaryDirectory() as d:
            previous = Path.cwd()
            try:
                os.chdir(d)
                enc = model(
                    [
                        h.make_node("Mul", ["audio", "scale"], ["encoded"]),
                        h.make_node("Identity", ["length"], ["encoded_length"]),
                    ],
                    [
                        h.make_tensor_value_info("audio", T.FLOAT, [1, 2, "T"]),
                        h.make_tensor_value_info("length", T.INT64, [1]),
                    ],
                    [
                        h.make_tensor_value_info("encoded", T.FLOAT, [1, 2, "T"]),
                        h.make_tensor_value_info("encoded_length", T.INT64, [1]),
                    ],
                    [nh.from_array(np.full((1, 2, 1), 2, np.float32), "scale")],
                )
                onnx.save(
                    enc,
                    "raw.onnx",
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="weights",
                    size_threshold=0,
                )
                prompt = model(
                    [
                        h.make_node(
                            "Cast", ["prompt_index"], ["float_prompt"], to=T.FLOAT
                        ),
                        h.make_node("Reshape", ["float_prompt", "shape"], ["bias"]),
                        h.make_node("Add", ["encoded", "bias"], ["output"]),
                    ],
                    [
                        h.make_tensor_value_info("encoded", T.FLOAT, [1, 2, "T"]),
                        h.make_tensor_value_info("prompt_index", T.INT64, [1]),
                    ],
                    [h.make_tensor_value_info("output", T.FLOAT, [1, 2, "T"])],
                    [nh.from_array(np.array([1, 1, 1], np.int64), "shape")],
                )
                onnx.save(prompt, "prompt.onnx")
                merge_prompt("raw.onnx", "prompt.onnx", "encoder.onnx", {"test": "yes"})
                merged = onnx.load("encoder.onnx", load_external_data=False)
                self.assertEqual(
                    [o.name for o in merged.graph.output],
                    ["prompt_output", "encoded_length"],
                )
                self.assertEqual(
                    [i.name for i in merged.graph.input],
                    ["audio", "length", "prompt_index"],
                )
                session = ort.InferenceSession(
                    "encoder.onnx", providers=["CPUExecutionProvider"]
                )
                x = np.ones((1, 2, 3), np.float32)
                for language in (0, 2):
                    output, length = session.run(
                        None,
                        dict(
                            audio=x,
                            length=np.array([3], np.int64),
                            prompt_index=np.array([language], np.int64),
                        ),
                    )
                    np.testing.assert_array_equal(output, x * 2 + language)
                    np.testing.assert_array_equal(length, [3])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
