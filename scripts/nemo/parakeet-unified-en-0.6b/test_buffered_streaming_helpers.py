#!/usr/bin/env python3
# Copyright      2026  Milan Leonard
import unittest

import numpy as np

from buffered_streaming_helpers import normalize_per_feature, slice_feature_buffer


class TestBufferedStreamingHelpers(unittest.TestCase):
    def test_slice_feature_buffer_starts_without_left_padding(self):
        features = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)

        window, valid_center_frames = slice_feature_buffer(
            features,
            center_start=0,
            left=4,
            chunk=3,
            right=2,
        )

        self.assertEqual(window.shape, (5, 2))
        np.testing.assert_array_equal(window, features[:5])
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

        self.assertEqual(window.shape, (5, 2))
        np.testing.assert_array_equal(window, features)
        self.assertEqual(valid_center_frames, 1)

    def test_final_window_retains_total_capacity(self):
        features = np.arange(800, dtype=np.float32).reshape(800, 1)
        window, valid_center = slice_feature_buffer(features, 734, 560, 56, 56)
        self.assertEqual(valid_center, 66)
        self.assertEqual(window.shape, (672, 1))
        np.testing.assert_array_equal(window, features[128:800])
        self.assertEqual(window.shape[0] - valid_center, 606)

    def test_short_stream_has_no_synthetic_context(self):
        features = np.arange(10, dtype=np.float32).reshape(10, 1)
        window, valid_center = slice_feature_buffer(features, 0, 560, 56, 56)
        self.assertEqual(valid_center, 10)
        np.testing.assert_array_equal(window, features)

    def test_final_chunk_when_right_context_exceeds_chunk(self):
        features = np.arange(18, dtype=np.float32).reshape(18, 1)
        window, valid_center = slice_feature_buffer(features, 12, 8, 2, 5)
        self.assertEqual(valid_center, 6)
        np.testing.assert_array_equal(window, features[3:18])

    def test_constant_log_floor_normalizes_to_zero(self):
        for frames in (112, 672):
            features = np.full((frames, 2), -15.94, dtype=np.float32)
            np.testing.assert_array_equal(
                normalize_per_feature(features), np.zeros_like(features)
            )

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
        np.testing.assert_allclose(
            normalized.std(axis=0, ddof=1), [1.0, 1.0], atol=2e-5
        )


if __name__ == "__main__":
    unittest.main()
