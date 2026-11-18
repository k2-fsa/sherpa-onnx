#!/usr/bin/env python3
# Copyright (c)  2026  Silvio Tomatis
"""Verify that invalid Sortformer models raise catchable exceptions.

Usage: PYTHONPATH=build/lib python3 test_model_validation.py model.onnx
"""

import sys
import tempfile
from pathlib import Path

import _sherpa_onnx as sherpa_onnx
import numpy as np
import onnx
from onnx import TensorProto, helper


def test_model_validation(model_path):
    metadata = {
        p.key: p.value
        for p in onnx.load(model_path, load_external_data=False).metadata_props
    }
    # Tiny graph with deliberately wrong output shapes, avoiding a copy of
    # the real model weights for each metadata validation case.
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                [name],
                value=helper.make_tensor(name, TensorProto.FLOAT, [1, 1, 1], [0]),
            )
            for name in ("probs", "chunk_embeds")
        ],
        "sortformer-validation",
        [
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, "T", 128]),
            helper.make_tensor_value_info(
                "cached_embeds", TensorProto.FLOAT, [1, "C", 512]
            ),
            helper.make_tensor_value_info("num_frames", TensorProto.INT64, []),
        ],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 1])
            for name in ("probs", "chunk_embeds")
        ],
    )
    cases = [
        ("model_type", None),
        ("model_type", "unsupported"),
        ("version", "1"),
        ("version", "bad"),
        ("version", "2147483648"),
        ("sample_rate", "0"),
        ("sample_rate", "-1"),
        ("preemphasis", None),
        ("preemphasis", ""),
        ("preemphasis", "bad"),
        ("preemphasis", "0.97junk"),
        ("preemphasis", "nan"),
        ("preemphasis", "inf"),
        ("preemphasis", "1e1000"),
        ("silence_embeds", None),
        ("silence_embeds", "bad"),
        ("silence_embeds", "0"),
        ("silence_embeds", ",".join(["nan"] * int(metadata["hidden_size"]))),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig()
        )
        config.segmentation.sortformer.model = str(path)
        config.segmentation.num_threads = 1
        for key, value in cases + [(None, None)]:
            props = dict(metadata)
            if key is not None:
                if value is None:
                    del props[key]
                else:
                    props[key] = value
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
            )
            helper.set_model_props(model, props)
            onnx.save(model, path)
            try:
                sd = sherpa_onnx.OfflineSpeakerDiarization(config)
                if key is None:
                    sd.process(np.zeros(16000, dtype=np.float32))
            except RuntimeError as error:
                message = str(error)
                assert message, (key, value)
                if key is None:
                    assert "output shapes/types" in message, message
                elif key in ("preemphasis", "silence_embeds"):
                    assert key in message, message
            else:
                raise AssertionError(f"Expected RuntimeError: {key}={value}")
    print(f"PASS: {len(cases)} invalid metadata cases and invalid output shapes")


if __name__ == "__main__":
    test_model_validation(sys.argv[1])
