#!/usr/bin/env python3
# Copyright (c) 2026 Julian Pscheid

"""Exercise the real unified runtime with small deterministic ONNX models."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import _sherpa_onnx as sherpa_onnx
except ImportError:
    from sherpa_onnx.lib import _sherpa_onnx as sherpa_onnx

try:
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh
except ImportError:
    onnx = None


def tensor(name, value):
    return nh.from_array(np.asarray(value), name=name)


def save_graph(path, nodes, inputs, outputs, initializers, metadata=None):
    model = h.make_model(
        h.make_graph(nodes, path.stem, inputs, outputs, initializers),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=8,
    )
    for key, value in (metadata or {}).items():
        model.metadata_props.add(key=key, value=str(value))
    onnx.checker.check_model(model)
    onnx.save(model, path)


def create_models(directory, window_length=False):
    # Only encoder frame zero emits a word. All subsequent frames emit blank.
    # This detects fixed left padding and accidental restarts of encoder context.
    meta = dict(
        vocab_size=4,
        normalize_type="",
        pred_rnn_layers=1,
        pred_hidden=1,
        subsampling_factor=8,
        model_type="EncDecRNNTBPEModel",
        streaming_model_type="nemo_parakeet_unified_streaming",
        left_feature_frames=560,
        chunk_feature_frames=56,
        right_feature_frames=56,
        left_encoder_frames=70,
        chunk_encoder_frames=7,
        right_encoder_frames=7,
        feat_dim=128,
    )
    enc_inputs = [
        h.make_tensor_value_info("audio", T.FLOAT, [1, 128, "T"]),
        h.make_tensor_value_info("length", T.INT64, [1]),
    ]
    init = [
        tensor("depth", np.array(5, np.int64)),
        tensor("values", np.array([0, 2], np.float32)),
        tensor("shape", np.array([1, 5, 1], np.int64)),
        tensor("round_up", np.array([7], np.int64)),
        tensor("factor", np.array([8], np.int64)),
    ]
    init.append(tensor("prompt_index", np.array([0], np.int64)))
    tail = np.zeros((1, 5, 199), np.float32)
    tail[:, 4, :] = 2
    init.append(tensor("tail", tail))
    nodes = [
        h.make_node("Add", ["length", "round_up"], ["rounded_length"]),
        h.make_node("Div", ["rounded_length", "factor"], ["out_length"]),
    ]
    prompt_name = "prompt_index"
    if window_length:
        init.append(tensor("capacity", np.array([672], np.int64)))
        nodes += [
            h.make_node("Equal", ["length", "capacity"], ["full"]),
            h.make_node("Not", ["full"], ["short"]),
            h.make_node("Cast", ["short"], ["word_index"], to=T.INT64),
        ]
        prompt_name = "word_index"
    nodes += [
        h.make_node("OneHot", [prompt_name, "depth", "values"], ["word"], axis=-1),
        h.make_node("Reshape", ["word", "shape"], ["first"]),
    ]
    if window_length:
        init.append(tensor("repeats", np.array([1, 1, 200], np.int64)))
        nodes.append(h.make_node("Tile", ["first", "repeats"], ["encoded"]))
    else:
        nodes.append(h.make_node("Concat", ["first", "tail"], ["encoded"], axis=2))
    save_graph(
        directory / "encoder.onnx",
        nodes,
        enc_inputs,
        [
            h.make_tensor_value_info("encoded", T.FLOAT, [1, 5, 200]),
            h.make_tensor_value_info("out_length", T.INT64, [1]),
        ],
        init,
        meta,
    )
    save_graph(
        directory / "decoder.onnx",
        [
            h.make_node("Equal", ["targets", "blank"], ["is_blank"]),
            h.make_node("Not", ["is_blank"], ["has_word"]),
            h.make_node("Cast", ["has_word"], ["value"], to=T.FLOAT),
            h.make_node("Reshape", ["value", "shape"], ["decoded"]),
            h.make_node("Identity", ["length"], ["out_length"]),
            h.make_node("Identity", ["h"], ["out_h"]),
            h.make_node("Identity", ["c"], ["out_c"]),
        ],
        [
            h.make_tensor_value_info("targets", T.INT32, [1, 1]),
            h.make_tensor_value_info("length", T.INT32, [1]),
            h.make_tensor_value_info("h", T.FLOAT, [1, 1, 1]),
            h.make_tensor_value_info("c", T.FLOAT, [1, 1, 1]),
        ],
        [
            h.make_tensor_value_info("decoded", T.FLOAT, [1, 1, 1]),
            h.make_tensor_value_info("out_length", T.INT32, [1]),
            h.make_tensor_value_info("out_h", T.FLOAT, [1, 1, 1]),
            h.make_tensor_value_info("out_c", T.FLOAT, [1, 1, 1]),
        ],
        [
            tensor("blank", np.array([[4]], np.int32)),
            tensor("shape", np.array([1, 1, 1], np.int64)),
        ],
        {"streaming_model_type": "nemo_parakeet_unified_streaming"},
    )
    # After emitting a token, force blank until the decoder is reset.
    mask = np.zeros((1, 5, 1), np.float32)
    mask[:, 4, :] = 100
    save_graph(
        directory / "joiner.onnx",
        [
            h.make_node("Mul", ["decoded", "mask"], ["blank_score"]),
            h.make_node("Add", ["encoded", "blank_score"], ["logits"]),
        ],
        [
            h.make_tensor_value_info("encoded", T.FLOAT, [1, 5, 1]),
            h.make_tensor_value_info("decoded", T.FLOAT, [1, 1, 1]),
        ],
        [h.make_tensor_value_info("logits", T.FLOAT, [1, 5, 1])],
        [tensor("mask", mask)],
    )
    (directory / "tokens.txt").write_text(
        "▁en 0\n▁de 1\n▁fr 2\n▁auto 3\n<blk> 4\n", encoding="utf-8"
    )


@unittest.skipIf(onnx is None, "onnx is needed to generate the test models")
class TestOnlineUnifiedRecognizer(unittest.TestCase):
    def create_recognizer(self, directory):
        transducer = sherpa_onnx.OnlineTransducerModelConfig(
            encoder=str(directory / "encoder.onnx"),
            decoder=str(directory / "decoder.onnx"),
            joiner=str(directory / "joiner.onnx"),
        )
        model = sherpa_onnx.OnlineModelConfig(
            transducer=transducer,
            tokens=str(directory / "tokens.txt"),
            num_threads=1,
        )
        config = sherpa_onnx.OnlineRecognizerConfig(
            feat_config=sherpa_onnx.FeatureExtractorConfig(),
            model_config=model,
            enable_endpoint=False,
            decoding_method="greedy_search",
        )
        return sherpa_onnx.OnlineRecognizer(config)

    def finish(self, recognizer, stream):
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        return recognizer.get_result(stream).text.strip()

    def test_english_encoder_without_prompt_input(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_models(directory)
            rec = self.create_recognizer(directory)
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.ones(20000, np.float32) * 0.1)
            self.assertEqual(self.finish(rec, stream), "en")

    def test_zero_lookahead_keeps_partial_hop(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_models(directory)
            encoder = onnx.load(directory / "encoder.onnx")
            meta = {m.key: m.value for m in encoder.metadata_props}
            meta.update(
                left_feature_frames=0,
                left_encoder_frames=0,
                right_feature_frames=0,
                right_encoder_frames=0,
            )
            # Emit only if the encoder sees energy from the final partial hop.
            # Exclude the extra masked feature frame, whose zeros are padding.
            tail = np.zeros((1, 5, 199), np.float32)
            tail[:, 4, :] = 2
            save_graph(
                directory / "encoder.onnx",
                [
                    h.make_node(
                        "Slice", ["audio", "zero", "length", "axis"], ["valid_audio"]
                    ),
                    h.make_node("ReduceMax", ["valid_audio"], ["peak"], keepdims=0),
                    h.make_node("Greater", ["peak", "threshold"], ["has_tail"]),
                    h.make_node("Unsqueeze", ["has_tail", "zero"], ["condition"]),
                    h.make_node(
                        "Where", ["condition", "word_id", "blank_id"], ["choice"]
                    ),
                    h.make_node(
                        "OneHot", ["choice", "depth", "values"], ["word"], axis=-1
                    ),
                    h.make_node("Reshape", ["word", "shape"], ["first"]),
                    h.make_node("Concat", ["first", "tail"], ["encoded"], axis=2),
                    h.make_node("Add", ["length", "round_up"], ["rounded"]),
                    h.make_node("Div", ["rounded", "factor"], ["out_length"]),
                ],
                list(encoder.graph.input),
                list(encoder.graph.output),
                [
                    tensor("zero", np.array([0], np.int64)),
                    tensor("axis", np.array([2], np.int64)),
                    tensor("threshold", np.array(-10, np.float32)),
                    tensor("word_id", np.array([0], np.int64)),
                    tensor("blank_id", np.array([4], np.int64)),
                    tensor("depth", np.array(5, np.int64)),
                    tensor("values", np.array([0, 2], np.float32)),
                    tensor("shape", np.array([1, 5, 1], np.int64)),
                    tensor("tail", tail),
                    tensor("round_up", np.array([7], np.int64)),
                    tensor("factor", np.array([8], np.int64)),
                ],
                meta,
            )
            rec = self.create_recognizer(directory)
            for drain_first in (False, True):
                for remainder in (0, 79):
                    with self.subTest(drain_first=drain_first, remainder=remainder):
                        audio = np.zeros(56 * 160 + remainder, np.float32)
                        audio[56 * 160 :] = 0.5
                        stream = rec.create_stream()
                        stream.accept_waveform(16000, audio)
                        if drain_first:
                            self.assertFalse(rec.is_ready(stream))
                            while rec.is_ready(stream):
                                rec.decode_stream(stream)
                        self.assertEqual(
                            self.finish(rec, stream), "en" if remainder else ""
                        )
                        self.assertFalse(rec.is_ready(stream))
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
            self.assertTrue(rec.is_ready(stream))

    def test_partial_hop_after_complete_step(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_models(directory)
            rec = self.create_recognizer(directory)
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.ones(112 * 160 + 1, np.float32) * 0.1)
            stream.input_finished()
            self.assertTrue(rec.is_ready(stream))
            rec.decode_stream(stream)
            self.assertTrue(rec.is_ready(stream))
            rec.decode_stream(stream)
            self.assertFalse(rec.is_ready(stream))

    def test_final_slice_with_partial_hop(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_models(directory)
            encoder = onnx.load(directory / "encoder.onnx")
            # Frame 83 lies beyond all regular seven-frame center slices.
            # A partial-hop EOF shifts left context from frame 77 to frame 76,
            # so the final eight-frame slice must include this marker.
            frames = np.zeros((1, 5, 200), np.float32)
            frames[:, 4, :] = 2
            frames[:, 4, 83] = 0
            frames[:, 0, 83] = 2
            save_graph(
                directory / "encoder.onnx",
                [
                    h.make_node("Identity", ["frames"], ["encoded"]),
                    h.make_node("Add", ["length", "round_up"], ["rounded"]),
                    h.make_node("Div", ["rounded", "factor"], ["out_length"]),
                ],
                list(encoder.graph.input),
                list(encoder.graph.output),
                [
                    tensor("frames", frames),
                    tensor("round_up", np.array([7], np.int64)),
                    tensor("factor", np.array([8], np.int64)),
                ],
                {m.key: m.value for m in encoder.metadata_props},
            )
            rec = self.create_recognizer(directory)
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.ones(672 * 160 + 79, np.float32) * 0.1)
            self.assertEqual(self.finish(rec, stream), "en")
            self.assertAlmostEqual(rec.get_result(stream).timestamps[0], 6.72, places=4)

    def test_reset_retains_predictor_context(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_models(directory, window_length=True)
            rec = self.create_recognizer(directory)
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.ones(20000, np.float32) * 0.1)
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            self.assertEqual(rec.get_result(stream).text.strip(), "de")
            rec.reset(stream)
            self.assertEqual(self.finish(rec, stream), "")

    def test_reset_retains_recurrent_states(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_models(directory, window_length=True)
            encoder = onnx.load(directory / "encoder.onnx")
            for value in encoder.graph.initializer:
                if value.name == "capacity":
                    value.CopyFrom(tensor("capacity", np.array([112], np.int64)))
            onnx.save(encoder, directory / "encoder.onnx")
            # g is the input recurrent counter; the next state increments it.
            save_graph(
                directory / "decoder.onnx",
                [
                    h.make_node("Identity", ["h"], ["decoded"]),
                    h.make_node("Identity", ["length"], ["out_length"]),
                    h.make_node("Add", ["h", "one"], ["out_h"]),
                    h.make_node("Add", ["c", "one"], ["out_c"]),
                ],
                [
                    h.make_tensor_value_info("targets", T.INT32, [1, 1]),
                    h.make_tensor_value_info("length", T.INT32, [1]),
                    h.make_tensor_value_info("h", T.FLOAT, [1, 1, 1]),
                    h.make_tensor_value_info("c", T.FLOAT, [1, 1, 1]),
                ],
                [
                    h.make_tensor_value_info("decoded", T.FLOAT, [1, 1, 1]),
                    h.make_tensor_value_info("out_length", T.INT32, [1]),
                    h.make_tensor_value_info("out_h", T.FLOAT, [1, 1, 1]),
                    h.make_tensor_value_info("out_c", T.FLOAT, [1, 1, 1]),
                ],
                [tensor("one", np.ones((1, 1, 1), np.float32))],
                {"streaming_model_type": "nemo_parakeet_unified_streaming"},
            )
            scores = np.eye(5, dtype=np.float32).reshape(5, 1, 5, 1) * 2
            save_graph(
                directory / "joiner.onnx",
                [
                    h.make_node("ArgMax", ["encoded"], ["phase"], axis=1, keepdims=1),
                    h.make_node("Equal", ["phase", "zero_id"], ["phase_zero"]),
                    h.make_node("Equal", ["phase", "one_id"], ["phase_one"]),
                    h.make_node("Equal", ["decoded", "zero"], ["g_zero"]),
                    h.make_node("Equal", ["decoded", "one"], ["g_one"]),
                    h.make_node("And", ["phase_zero", "g_zero"], ["emit_en"]),
                    h.make_node("And", ["phase_one", "g_one"], ["emit_de"]),
                    h.make_node("And", ["phase_one", "g_zero"], ["emit_error"]),
                    h.make_node("Where", ["emit_error", "fr", "blank"], ["fallback"]),
                    h.make_node("Where", ["emit_de", "de", "fallback"], ["next"]),
                    h.make_node("Where", ["emit_en", "en", "next"], ["logits"]),
                ],
                [
                    h.make_tensor_value_info("encoded", T.FLOAT, [1, 5, 1]),
                    h.make_tensor_value_info("decoded", T.FLOAT, [1, 1, 1]),
                ],
                [h.make_tensor_value_info("logits", T.FLOAT, [1, 5, 1])],
                [
                    tensor("zero_id", np.zeros((1, 1, 1), np.int64)),
                    tensor("one_id", np.ones((1, 1, 1), np.int64)),
                    tensor("zero", np.zeros((1, 1, 1), np.float32)),
                    tensor("one", np.ones((1, 1, 1), np.float32)),
                    tensor("en", scores[0]),
                    tensor("de", scores[1]),
                    tensor("fr", scores[2]),
                    tensor("blank", scores[4]),
                ],
            )
            rec = self.create_recognizer(directory)
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.ones(20000, np.float32) * 0.1)
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            self.assertEqual(rec.get_result(stream).text.strip(), "en")
            rec.reset(stream)
            self.assertEqual(self.finish(rec, stream), "de")


if __name__ == "__main__":
    unittest.main()
