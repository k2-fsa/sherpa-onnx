#!/usr/bin/env python3
# Copyright (c)  2026  Silvio Tomatis
"""Test exported Nemotron models through the native Python binding.

Set PYTHONPATH to the CMake build's lib directory. No embedding model or
clustering configuration is supplied. FP32 segments are compared with the
independent NumPy pipeline in test_onnx.py; INT8 is tested separately because
quantization changes predictions.
"""

import argparse
from pathlib import Path

import _sherpa_onnx as sherpa_onnx
import numpy as np
import soundfile as sf

from test_onnx import OnnxModel, compute_features, diarize, to_segments


def segments(diarizer, samples):
    result = diarizer.process(samples).sort_by_start_time()
    ans = [(s.start, s.end, s.speaker) for s in result]
    for start, end, speaker in ans:
        assert np.isfinite([start, end]).all(), ans
        assert 0 <= start < end <= len(samples) / 16000 + 1e-4, ans
        assert 0 <= speaker < 8, ans
    return ans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    args = parser.parse_args()
    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    assert sample_rate == 16000 and audio.shape[1] == 1
    audio = audio[:, 0]
    assert len(audio) > 435360, "The fixture must cross a full chunk boundary"

    for name in ("model.onnx", "model.int8.onnx"):
        model = args.model_dir / name
        fp32 = name == "model.onnx"
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                sortformer=sherpa_onnx.OfflineSpeakerSegmentationSortformerModelConfig(
                    str(model), 0.5
                ),
                num_threads=2,
            ),
            # Without duration filtering, every FP32 segment can be compared
            # directly with the NumPy pipeline's activity decisions.
            min_duration_on=0 if fp32 else 0.3,
            min_duration_off=0 if fp32 else 0.5,
        )
        assert config.validate(), str(config)
        sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        assert sd.sample_rate == 16000
        reference = OnnxModel(str(model)) if fp32 else None

        # Empty/sub-frame input, around the exact 8-frame boundary, around
        # the exact 340-embedding chunk boundary, and multiple chunks.
        for size in (0, 1, 1279, 1280, 435040, 435200, 435360, len(audio)):
            samples = audio[:size]
            actual = segments(sd, samples)
            assert actual == segments(sd, samples), "Cache leaked between calls"
            if size < 160:
                assert actual == [], actual
            if size == len(audio):
                assert {s[2] for s in actual} == {0, 1, 2, 3}, actual
            if fp32:
                expected = []
                if size >= 160:
                    features = compute_features(samples, reference)
                    probs = diarize(reference, features)
                    expected = to_segments(probs, 0.5, 0.01)
                assert len(actual) == len(expected), (size, actual, expected)
                for a, e in zip(actual, expected):
                    assert a[2] == e[2], (size, a, e)
                    # Segment timestamps are stored as float32 in C++.
                    np.testing.assert_allclose(a[:2], e[:2], rtol=0, atol=1e-4)
            print(f"PASS {name}: {size} samples, {len(actual)} segments", flush=True)


if __name__ == "__main__":
    main()
