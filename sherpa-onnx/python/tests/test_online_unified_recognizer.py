#!/usr/bin/env python3
# Copyright (c) 2026 Julian Pscheid

"""Exercise the real unified runtime with small deterministic ONNX models."""

import json
import subprocess
import sys
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


DECODING_METHODS = ("greedy_search", "modified_beam_search")


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


def create_beam_models(directory):
    # The first two chunks favor A and B C respectively. A stays ahead of
    # B after chunk one (.60 vs .39), but .60 * .51 < .39 in chunk two.
    # Phase two emits D only from the predictor state after C.
    probabilities = blank_probabilities(4, 5)
    probabilities[0, 4] = [0.60, 0.39, 1e-6, 2e-6, 0.01]
    probabilities[1, 0] = [1e-6, 2e-6, 3e-6, 0.49, 0.51]
    probabilities[1, 1, [2, 4]] = [1, 5e-6]
    probabilities[2, 2, [3, 4]] = [1, 5e-6]
    phases = np.full((1, 1, 200), 3, np.float32)
    phases[0, 0, [0, 7, 14]] = [0, 1, 2]
    create_lookup_models(directory, phases, probabilities)


def blank_probabilities(num_phases, num_states):
    return np.tile(
        np.array([1e-6, 2e-6, 3e-6, 4e-6, 1], np.float32),
        (num_phases, num_states, 1),
    )


def create_lookup_models(
    directory, phases, probabilities, recurrent=False, logit_rows=()
):
    create_models(directory)
    encoder = onnx.load(directory / "encoder.onnx")
    metadata = {entry.key: entry.value for entry in encoder.metadata_props}
    metadata.update(right_feature_frames=0, right_encoder_frames=0)
    save_graph(
        directory / "encoder.onnx",
        [
            h.make_node("Identity", ["phases"], ["encoded"]),
            h.make_node("Add", ["length", "round_up"], ["rounded"]),
            h.make_node("Div", ["rounded", "factor"], ["out_length"]),
        ],
        list(encoder.graph.input),
        [
            h.make_tensor_value_info("encoded", T.FLOAT, list(phases.shape)),
            h.make_tensor_value_info("out_length", T.INT64, [1]),
        ],
        [
            tensor("phases", phases),
            tensor("round_up", np.array([7], np.int64)),
            tensor("factor", np.array([8], np.int64)),
        ],
        metadata,
    )
    decoder = onnx.load(directory / "decoder.onnx")
    if recurrent:
        # The output is the counter before consuming this predictor input.
        decoder_nodes = [
            h.make_node("Add", ["h", "c"], ["sum"]),
            h.make_node("Mul", ["sum", "half"], ["decoded"]),
            h.make_node("Add", ["h", "one"], ["out_h"]),
            h.make_node("Add", ["c", "one"], ["out_c"]),
        ]
        initializers = [
            tensor("half", np.array(0.5, np.float32)),
            tensor("one", np.array(1, np.float32)),
        ]
    else:
        decoder_nodes = [
            h.make_node("Cast", ["targets"], ["token"], to=T.FLOAT),
            h.make_node("Reshape", ["token", "shape"], ["decoded"]),
            h.make_node("Identity", ["h"], ["out_h"]),
            h.make_node("Identity", ["c"], ["out_c"]),
        ]
        initializers = [tensor("shape", np.array([1, 1, 1], np.int64))]
    decoder_nodes.insert(1, h.make_node("Identity", ["length"], ["out_length"]))
    save_graph(
        directory / "decoder.onnx",
        decoder_nodes,
        list(decoder.graph.input),
        list(decoder.graph.output),
        initializers,
        {entry.key: entry.value for entry in decoder.metadata_props},
    )
    normalized = probabilities / probabilities.sum(axis=-1, keepdims=True)
    logits = np.log(normalized)
    for phase, state, row in logit_rows:
        logits[phase, state] = row
    save_graph(
        directory / "joiner.onnx",
        [
            h.make_node("Cast", ["encoded"], ["phase"], to=T.INT64),
            h.make_node("Reshape", ["phase", "scalar"], ["phase_index"]),
            h.make_node("Gather", ["table", "phase_index"], ["phase_logits"]),
            h.make_node("Cast", ["decoded"], ["state"], to=T.INT64),
            h.make_node("Reshape", ["state", "scalar"], ["state_index"]),
            h.make_node("Gather", ["phase_logits", "state_index"], ["logits"]),
        ],
        [
            h.make_tensor_value_info("encoded", T.FLOAT, [1, 1, 1]),
            h.make_tensor_value_info("decoded", T.FLOAT, [1, 1, 1]),
        ],
        [h.make_tensor_value_info("logits", T.FLOAT, [1, 5, 1])],
        [
            tensor("scalar", np.array([], np.int64)),
            tensor("table", logits[:, :, np.newaxis, :, np.newaxis]),
        ],
    )
    (directory / "tokens.txt").write_text(
        "▁A 0\n▁B 1\n▁C 2\n▁D 3\n<blk> 4\n", encoding="utf-8"
    )


@unittest.skipIf(onnx is None, "onnx is needed to generate the test models")
class TestOnlineUnifiedRecognizer(unittest.TestCase):
    def create_recognizer(
        self,
        directory,
        decoding_method="greedy_search",
        enable_endpoint=False,
        **options,
    ):
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
            enable_endpoint=enable_endpoint,
            decoding_method=decoding_method,
            **options,
        )
        return sherpa_onnx.OnlineRecognizer(config)

    def finish(self, recognizer, stream):
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        return recognizer.get_result(stream).text.strip()

    def decode_runner_up(self, directory, method, beam):
        rec = self.create_recognizer(directory, method, max_active_paths=beam)
        stream = rec.create_stream()
        stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
        self.assertTrue(rec.is_ready(stream))
        rec.decode_stream(stream)
        self.assertEqual(rec.get_result(stream).text.strip(), "A")
        self.assertFalse(rec.is_ready(stream))
        stream.accept_waveform(16000, np.zeros(55 * 160, np.float32))
        return self.finish(rec, stream)

    def test_beam_preserves_runner_up_across_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_beam_models(directory)
            self.assertEqual(self.decode_runner_up(directory, "greedy_search", 2), "A")
            for beam in (2, 4, 8):
                with self.subTest(beam=beam):
                    self.assertEqual(
                        self.decode_runner_up(directory, "modified_beam_search", beam),
                        "B C",
                    )

    def test_beam_combines_alignments_for_same_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            probabilities = blank_probabilities(2, 5)
            probabilities[0, 4, [0, 4]] = [0.33, 0.67]
            for second_frame in (1, 7):
                with self.subTest(second_frame=second_frame):
                    # Empty has probability .67**2. Neither alignment of A
                    # wins alone, but together they have .33 + .67 * .33.
                    phases = np.ones((1, 1, 200), np.float32)
                    phases[0, 0, [0, second_frame]] = 0
                    create_lookup_models(directory, phases, probabilities)
                    rec = self.create_recognizer(
                        directory, "modified_beam_search", max_active_paths=8
                    )
                    stream = rec.create_stream()
                    stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
                    self.assertTrue(rec.is_ready(stream))
                    rec.decode_stream(stream)
                    if second_frame == 7:
                        self.assertEqual(rec.get_result(stream).text.strip(), "")
                    stream.accept_waveform(16000, np.zeros(55 * 160, np.float32))
                    self.assertEqual(self.finish(rec, stream), "A")
                    result = rec.get_result(stream)
                    self.assertEqual(result.timestamps, [0.0])
                    self.assertEqual(len(result.ys_probs), 1)

    def test_beam_metadata_uses_strongest_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            probabilities = blank_probabilities(3, 5)
            probabilities[0, 4, [0, 4]] = [0.22, 0.78]
            probabilities[1, 4, [0, 4]] = [0.60, 0.40]
            # The first two A alignments sum to .3916. The third alignment
            # has probability .36504 and is the strongest individual path.
            phases = np.full((1, 1, 200), 2, np.float32)
            phases[0, 0, [0, 1, 7]] = [0, 0, 1]
            create_lookup_models(directory, phases, probabilities)
            rec = self.create_recognizer(
                directory, "modified_beam_search", max_active_paths=8
            )
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
            self.assertTrue(rec.is_ready(stream))
            rec.decode_stream(stream)
            self.assertEqual(rec.get_result(stream).text.strip(), "")
            stream.accept_waveform(16000, np.zeros(55 * 160, np.float32))
            self.assertEqual(self.finish(rec, stream), "A")
            result = rec.get_result(stream)
            np.testing.assert_allclose(result.timestamps, [0.56], atol=1e-6)
            np.testing.assert_allclose(result.ys_probs, [np.log(0.60)], atol=1e-4)

    def test_beam_one_matches_greedy(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_beam_models(directory)
            greedy = self.decode_runner_up(directory, "greedy_search", 1)
            self.assertEqual(greedy, "A")
            self.assertEqual(
                self.decode_runner_up(directory, "modified_beam_search", 1), greedy
            )

    def test_beam_one_preserves_close_logit_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            phases = np.ones((1, 1, 200), np.float32)
            phases[0, 0, 0] = 0
            for row, expected in (([0, 1e-9], "B"), ([1e-9, 0], "A")):
                with self.subTest(expected=expected):
                    # The order is distinct before float32 log-softmax, which
                    # rounds both leading scores to the same value.
                    create_lookup_models(
                        directory,
                        phases,
                        blank_probabilities(2, 5),
                        logit_rows=[(0, 4, row + [-10, -10, -10])],
                    )
                    for method in DECODING_METHODS:
                        with self.subTest(method=method):
                            rec = self.create_recognizer(
                                directory, method, max_active_paths=1
                            )
                            stream = rec.create_stream()
                            stream.accept_waveform(
                                16000, np.zeros(56 * 160, np.float32)
                            )
                            self.assertEqual(self.finish(rec, stream), expected)

    def test_unified_beam_budget_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_beam_models(directory)
            self.assertEqual(
                self.decode_runner_up(directory, "modified_beam_search", 32), "B C"
            )

    def assert_rejected(self, directory, option, value, diagnostic):
        # Native configuration errors terminate the calling process.
        code = """
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
from test_online_unified_recognizer import TestOnlineUnifiedRecognizer, sherpa_onnx

directory = Path(sys.argv[2])
option, value = sys.argv[3:]
converters = {"max_active_paths": int, "blank_penalty": float, "hotwords_file": str}
options = {}
if option in converters:
    options[option] = converters[option](value)
elif option == "lm":
    options["lm_config"] = sherpa_onnx.OnlineLMConfig(model=value)
elif option == "lodr_fst":
    options["lm_config"] = sherpa_onnx.OnlineLMConfig(lodr_fst=value)
test = TestOnlineUnifiedRecognizer()
rec = test.create_recognizer(directory, "modified_beam_search", **options)
if option == "stream_hotwords":
    rec.create_stream(value)
elif option == "joiner_output":
    stream = rec.create_stream()
    stream.accept_waveform(16000, np.zeros(20000, np.float32))
    test.finish(rec, stream)
"""
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(Path(__file__).resolve().parent),
                    str(directory),
                    option,
                    value,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        failure = caught.exception
        output = failure.stdout + failure.stderr
        self.assertIn(diagnostic, output)
        self.assertNotIn("synthetic-hotword", output)

    def test_unified_beam_rejects_invalid_options(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_models(directory)
            hotwords = directory / "hotwords.txt"
            hotwords.write_text("synthetic-hotword\n", encoding="utf-8")
            cases = [
                ("max_active_paths", "0", "max_active_paths in [1, 32]"),
                ("max_active_paths", "-1", "max_active_paths in [1, 32]"),
                ("max_active_paths", "33", "max_active_paths in [1, 32]"),
                ("blank_penalty", "nan", "finite blank_penalty"),
                ("blank_penalty", "inf", "finite blank_penalty"),
                ("blank_penalty", "-inf", "finite blank_penalty"),
                ("lm", str(directory / "encoder.onnx"), "LM fusion is not supported"),
                ("lodr_fst", str(directory / "lodr.fst"), "LM fusion is not supported"),
                ("hotwords_file", str(hotwords), "Hotwords are not supported"),
                ("stream_hotwords", "synthetic-hotword", "Hotwords are not supported"),
            ]
            for option, value, diagnostic in cases:
                with self.subTest(option=option, value=value):
                    self.assert_rejected(directory, option, value, diagnostic)

    def test_unified_beam_rejects_invalid_joiner_output(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_models(directory)
            joiner = onnx.load(directory / "joiner.onnx")
            invalid_logits = (
                np.zeros((1, 4, 1), np.float32),
                np.zeros((1, 5, 1), np.int64),
            )
            for logits in invalid_logits:
                with self.subTest(shape=logits.shape, dtype=logits.dtype):
                    value = tensor("bad_logits", logits)
                    output = h.make_tensor_value_info(
                        "logits", value.data_type, list(logits.shape)
                    )
                    save_graph(
                        directory / "joiner.onnx",
                        [h.make_node("Identity", ["bad_logits"], ["logits"])],
                        list(joiner.graph.input),
                        [output],
                        [value],
                    )
                    self.assert_rejected(
                        directory,
                        "joiner_output",
                        "",
                        "Invalid Parakeet Unified joiner output",
                    )

    def test_beam_discards_nan_candidate_when_valid_path_remains(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            probabilities = blank_probabilities(3, 5)
            probabilities[0, 4, [0, 4]] = [0.45, 0.55]
            probabilities[1, 4, [1, 4]] = [0.39, 0.61]
            phases = np.full((1, 1, 200), 2, np.float32)
            phases[0, 0, :2] = [0, 1]
            create_lookup_models(
                directory,
                phases,
                probabilities,
                logit_rows=[(1, 0, [0, np.nan, 0, 0, 0])],
            )
            rec = self.create_recognizer(
                directory, "modified_beam_search", max_active_paths=2
            )
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
            self.assertTrue(rec.is_ready(stream))
            rec.decode_stream(stream)
            # Skipping the invalid row at zero cost would let A keep ~.45,
            # beating the empty path after its valid blank (~.55 * .61).
            self.assertEqual(rec.get_result(stream).tokens, [])

    def test_beam_advances_when_all_joiner_rows_are_nan(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            probabilities = blank_probabilities(3, 5)
            probabilities[1, 4, [2, 4]] = [1, 5e-6]
            phases = np.full((1, 1, 200), 2, np.float32)
            phases[0, 0, 6:8] = [0, 1]
            create_lookup_models(
                directory,
                phases,
                probabilities,
                logit_rows=[
                    (0, state, [10, np.nan, -10, -10, 0]) for state in range(5)
                ],
            )
            rec = self.create_recognizer(
                directory,
                "modified_beam_search",
                enable_endpoint=True,
                max_active_paths=2,
                endpoint_config=sherpa_onnx.EndpointConfig(
                    rule1=sherpa_onnx.EndpointRule(False, 0.52, 0),
                ),
            )
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
            self.assertTrue(rec.is_ready(stream))
            rec.decode_stream(stream)
            result = rec.get_result(stream)
            self.assertEqual(result.tokens, [])
            # Six valid blank frames (0.48 s) do not meet the threshold.
            # Counting the corrupt seventh frame would trigger an endpoint.
            self.assertFalse(rec.is_endpoint(stream))
            self.assertEqual(
                json.loads(result.as_json_string())["num_trailing_blanks"], 0
            )
            stream.accept_waveform(16000, np.zeros(55 * 160, np.float32))
            self.assertEqual(self.finish(rec, stream), "C")
            result = rec.get_result(stream)
            self.assertAlmostEqual(result.timestamps[0], 0.56)
            self.assertTrue(np.all(np.isfinite(result.ys_probs)))

    def test_beam_recovers_nan_when_other_paths_have_zero_probability(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            phases = np.full((1, 1, 200), 2, np.float32)
            phases[0, 0, :2] = [0, 1]
            create_lookup_models(
                directory,
                phases,
                blank_probabilities(3, 5),
                logit_rows=[
                    (0, 4, [-np.inf] * 4 + [0]),
                    (1, 4, [0, np.nan, 0, 0, 0]),
                ],
            )
            rec = self.create_recognizer(
                directory, "modified_beam_search", max_active_paths=2
            )
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
            self.assertTrue(rec.is_ready(stream))
            rec.decode_stream(stream)
            # The first frame permits only blank. Any token path that reaches
            # the NaN at the second frame has zero probability.
            self.assertEqual(rec.get_result(stream).tokens, [])

    def test_beam_reset_commits_reported_winner(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_beam_models(directory)
            rec = self.create_recognizer(
                directory, "modified_beam_search", max_active_paths=2
            )
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.zeros(113 * 160, np.float32))
            rec.decode_stream(stream)
            rec.decode_stream(stream)
            before = rec.get_result(stream)
            self.assertEqual(before.text.strip(), "B C")
            self.assertEqual(before.segment, 0)
            np.testing.assert_allclose(before.timestamps, [0, 0.56], atol=1e-6)
            rec.reset(stream)
            empty = rec.get_result(stream)
            self.assertEqual(empty.text.strip(), "")
            self.assertEqual(empty.tokens, [])
            self.assertEqual(empty.timestamps, [])
            self.assertEqual(empty.ys_probs, [])
            self.assertEqual(empty.segment, 1)
            self.assertAlmostEqual(empty.start_time, 1.12)

            rec.reset(stream)
            self.assertEqual(rec.get_result(stream).segment, 1)
            stream.accept_waveform(16000, np.zeros(55 * 160, np.float32))
            self.assertEqual(self.finish(rec, stream), "D")
            after = rec.get_result(stream)
            self.assertEqual(after.timestamps, [0.0])
            self.assertAlmostEqual(after.start_time, empty.start_time)

    def test_beam_symbol_cap_consumes_pending_token_once(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            probabilities = blank_probabilities(3, 16)
            probabilities[0, :10, 0] = 1
            probabilities[0, :10, 4] = 5e-6
            probabilities[1, 10, [2, 4]] = [1, 5e-6]
            phases = np.full((1, 1, 200), 2, np.float32)
            phases[0, 0, 6:8] = [0, 1]
            create_lookup_models(directory, phases, probabilities, recurrent=True)
            rec = self.create_recognizer(
                directory, "modified_beam_search", max_active_paths=1
            )
            ten_tokens = " ".join(["A"] * 10)
            for reset in (False, True):
                with self.subTest(reset=reset):
                    stream = rec.create_stream()
                    stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
                    self.assertTrue(rec.is_ready(stream))
                    rec.decode_stream(stream)
                    result = rec.get_result(stream)
                    self.assertEqual(result.text.strip(), ten_tokens)
                    np.testing.assert_allclose(result.timestamps, [0.48] * 10)
                    self.assertEqual(
                        json.loads(result.as_json_string())["num_trailing_blanks"], 0
                    )
                    if reset:
                        rec.reset(stream)
                    stream.accept_waveform(16000, np.zeros(55 * 160, np.float32))
                    self.assertEqual(
                        self.finish(rec, stream), "C" if reset else ten_tokens + " C"
                    )

    def test_beam_multi_symbol_frame_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            probabilities = blank_probabilities(2, 5)
            probabilities[0, 4, [2, 4]] = [1, 5e-6]
            probabilities[0, 2, [3, 4]] = [1, 5e-6]
            phases = np.ones((1, 1, 200), np.float32)
            phases[0, 0, 0] = 0
            create_lookup_models(directory, phases, probabilities)
            rec = self.create_recognizer(directory, "modified_beam_search")
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.zeros(56 * 160, np.float32))
            self.assertEqual(self.finish(rec, stream), "C D")
            result = rec.get_result(stream)
            self.assertEqual(len(result.tokens), 2)
            self.assertEqual(result.timestamps, [0, 0])
            self.assertEqual(len(result.ys_probs), 2)
            self.assertTrue(np.all(np.isfinite(result.ys_probs)))

    def test_beam_empty_and_repeated_reset(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            phases = np.zeros((1, 1, 200), np.float32)
            create_lookup_models(directory, phases, blank_probabilities(1, 5))
            rec = self.create_recognizer(directory, "modified_beam_search")
            stream = rec.create_stream()
            rec.reset(stream)
            rec.reset(stream)
            result = rec.get_result(stream)
            self.assertEqual(result.segment, 0)
            self.assertEqual(result.start_time, 0.0)
            stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
            self.assertTrue(rec.is_ready(stream))
            rec.decode_stream(stream)
            rec.reset(stream)
            rec.reset(stream)
            result = rec.get_result(stream)
            self.assertEqual(result.segment, 0)
            self.assertAlmostEqual(result.start_time, 0.56)
            self.assertEqual(self.finish(rec, stream), "")

    def test_beam_blank_only_and_empty_input(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            phases = np.zeros((1, 1, 200), np.float32)
            create_lookup_models(directory, phases, blank_probabilities(1, 5))
            rec = self.create_recognizer(
                directory, "modified_beam_search", enable_endpoint=True
            )
            for seconds in (0, 8):
                with self.subTest(seconds=seconds):
                    stream = rec.create_stream()
                    if seconds:
                        stream.accept_waveform(
                            16000, np.zeros(seconds * 16000, np.float32)
                        )
                    self.assertEqual(self.finish(rec, stream), "")
                    self.assertFalse(rec.is_ready(stream))
                    result = rec.get_result(stream)
                    self.assertEqual(result.tokens, [])
                    self.assertEqual(result.timestamps, [])
                    self.assertEqual(result.ys_probs, [])
                    self.assertEqual(
                        json.loads(result.as_json_string())["num_trailing_blanks"],
                        100 if seconds else 0,
                    )
                    self.assertEqual(rec.is_endpoint(stream), bool(seconds))

    def test_beam_with_blank_penalty_rejects_weak_token_emissions(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            probabilities = blank_probabilities(2, 5)
            probabilities[0, 4, [0, 1, 4]] = [0.20, 0.06, 0.74]
            phases = np.ones((1, 1, 200), np.float32)
            phases[0, 0, [0, 7]] = 0
            create_lookup_models(directory, phases, probabilities)
            rec = self.create_recognizer(
                directory,
                "modified_beam_search",
                max_active_paths=8,
                blank_penalty=0.25,
            )
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.zeros(57 * 160, np.float32))
            self.assertTrue(rec.is_ready(stream))
            rec.decode_stream(stream)
            stream.accept_waveform(16000, np.zeros(55 * 160, np.float32))
            # After the blank penalty, the empty transcript has mass ~.475.
            # The two alignments for A combine to ~.404 and stay below it.
            self.assertEqual(self.finish(rec, stream), "")
            result = rec.get_result(stream)
            self.assertEqual(result.tokens, [])
            self.assertEqual(
                json.loads(result.as_json_string())["num_trailing_blanks"], 14
            )

    def test_beam_endpoint_after_trailing_silence(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_beam_models(directory)
            rec = self.create_recognizer(
                directory, "modified_beam_search", enable_endpoint=True
            )
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.full(169 * 160, 0.1, np.float32))
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            result = rec.get_result(stream)
            self.assertEqual(result.text.strip(), "B C D")
            self.assertFalse(rec.is_endpoint(stream))
            stream.accept_waveform(16000, np.zeros(8 * 16000, np.float32))
            self.assertEqual(self.finish(rec, stream), "B C D")
            after = rec.get_result(stream)
            self.assertEqual(after.timestamps, result.timestamps)
            self.assertGreaterEqual(
                json.loads(after.as_json_string())["num_trailing_blanks"], 100
            )
            self.assertTrue(rec.is_endpoint(stream))

    def test_beam_chunking_and_stream_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_beam_models(directory)
            rec = self.create_recognizer(directory, "modified_beam_search")
            short_audio = np.zeros(112 * 160, np.float32)
            long_audio = np.zeros(168 * 160, np.float32)
            rng = np.random.default_rng(4131)

            def decode(audio, packet_size):
                stream = rec.create_stream()
                start = 0
                while start < len(audio):
                    size = packet_size if packet_size else int(rng.integers(97, 2001))
                    stream.accept_waveform(16000, audio[start : start + size])
                    start += size
                    while rec.is_ready(stream):
                        rec.decode_stream(stream)
                self.finish(rec, stream)
                return rec.get_result(stream)

            def signature(result):
                return (
                    result.text.strip(),
                    result.tokens,
                    result.timestamps,
                    result.ys_probs,
                )

            short_result = decode(short_audio, len(short_audio))
            long_result = decode(long_audio, len(long_audio))
            self.assertEqual(short_result.text.strip(), "B C")
            self.assertEqual(long_result.text.strip(), "B C D")
            for packet_size in (823, None):
                with self.subTest(packet_size=packet_size):
                    self.assertEqual(
                        signature(decode(long_audio, packet_size)),
                        signature(long_result),
                    )

            short = rec.create_stream()
            long = rec.create_stream()
            split = 57 * 160
            short.accept_waveform(16000, short_audio[:split])
            long.accept_waveform(16000, long_audio)
            rec.decode_streams([short, long])
            rec.decode_streams([long])
            self.assertEqual(rec.get_result(short).text.strip(), "A")
            self.assertEqual(rec.get_result(long).text.strip(), "B C")
            short.accept_waveform(16000, short_audio[split:])
            short.input_finished()
            long.input_finished()
            ready = [stream for stream in (long, short) if rec.is_ready(stream)]
            while ready:
                rec.decode_streams(ready)
                ready = [stream for stream in ready if rec.is_ready(stream)]
            self.assertEqual(signature(rec.get_result(short)), signature(short_result))
            self.assertEqual(signature(rec.get_result(long)), signature(long_result))

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
            for method in DECODING_METHODS:
                with self.subTest(method=method):
                    rec = self.create_recognizer(directory, method, max_active_paths=1)
                    for remainder in (0, 1, 79):
                        with self.subTest(remainder=remainder):
                            audio = np.zeros(56 * 160 + remainder, np.float32)
                            audio[56 * 160 :] = 0.5
                            stream = rec.create_stream()
                            stream.accept_waveform(16000, audio)
                            self.assertFalse(rec.is_ready(stream))
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
            audio = np.full(112 * 160 + 1, 0.1, np.float32)
            for method in DECODING_METHODS:
                with self.subTest(method=method):
                    rec = self.create_recognizer(directory, method)
                    stream = rec.create_stream()
                    stream.accept_waveform(16000, audio)
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
            for method in DECODING_METHODS:
                with self.subTest(method=method):
                    rec = self.create_recognizer(directory, method, max_active_paths=1)
                    for remainder in (0, 1, 79):
                        with self.subTest(remainder=remainder):
                            stream = rec.create_stream()
                            audio = np.full(672 * 160 + remainder, 0.1, np.float32)
                            stream.accept_waveform(16000, audio)
                            self.assertEqual(self.finish(rec, stream), "en")
                            self.assertAlmostEqual(
                                rec.get_result(stream).timestamps[0],
                                6.72 if remainder else 6.64,
                                places=4,
                            )

    def test_reset_retains_predictor_context(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            create_models(directory, window_length=True)
            for method in DECODING_METHODS:
                with self.subTest(method=method):
                    rec = self.create_recognizer(directory, method, max_active_paths=1)
                    stream = rec.create_stream()
                    stream.accept_waveform(16000, np.full(20000, 0.1, np.float32))
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
            for method in DECODING_METHODS:
                with self.subTest(method=method):
                    rec = self.create_recognizer(directory, method, max_active_paths=1)
                    stream = rec.create_stream()
                    stream.accept_waveform(16000, np.full(20000, 0.1, np.float32))
                    while rec.is_ready(stream):
                        rec.decode_stream(stream)
                    self.assertEqual(rec.get_result(stream).text.strip(), "en")
                    rec.reset(stream)
                    self.assertEqual(self.finish(rec, stream), "de")


if __name__ == "__main__":
    unittest.main()
