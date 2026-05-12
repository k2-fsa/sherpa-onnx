#!/usr/bin/env python3
# Copyright      2026  Milan Leonard
import unittest

import numpy as np

from buffered_streaming_helpers import normalize_per_feature, slice_feature_buffer


class TestBufferedStreamingHelpers(unittest.TestCase):
    def test_slice_feature_buffer_zero_pads_left_and_right_context(self):
        features = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)

        window, valid_center_frames = slice_feature_buffer(
            features,
            center_start=0,
            left=4,
            chunk=3,
            right=2,
        )

        self.assertEqual(window.shape, (9, 2))
        np.testing.assert_array_equal(window[:4], np.zeros((4, 2), dtype=np.float32))
        np.testing.assert_array_equal(window[4:9], features[:5])
        self.assertEqual(valid_center_frames, 3)

    def test_slice_feature_buffer_reports_short_final_center_chunk(self):
        features = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)

        window, valid_center_frames = slice_feature_buffer(
            features,
            center_start=4,
            left=2,
            chunk=3,
            right=2,
        )

        self.assertEqual(window.shape, (7, 2))
        np.testing.assert_array_equal(window[:3], features[2:5])
        np.testing.assert_array_equal(window[3:], np.zeros((4, 2), dtype=np.float32))
        self.assertEqual(valid_center_frames, 1)

    def test_normalize_per_feature_normalizes_each_column(self):
        features = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ],
            dtype=np.float32,
        )

        normalized = normalize_per_feature(features)

        np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=0), [1.0, 1.0], atol=2e-5)


if __name__ == "__main__":
    unittest.main()
