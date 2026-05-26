#!/usr/bin/env python3
# Copyright      2026  Milan Leonard

import numpy as np


def normalize_per_feature(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-5
    return ((features - mean) / std).astype(np.float32)


def slice_feature_buffer(
    features: np.ndarray,
    center_start: int,
    left: int,
    chunk: int,
    right: int,
):
    total = left + chunk + right
    left_start = center_start - left
    right_end = center_start + chunk + right
    pad_left = max(0, -left_start)
    pad_right = max(0, right_end - features.shape[0])
    start = max(0, left_start)
    end = min(features.shape[0], right_end)

    window = features[start:end]
    if pad_left or pad_right:
        window = np.pad(window, ((pad_left, pad_right), (0, 0)), mode="constant")
    if window.shape[0] != total:
        raise ValueError(f"Expected {total} frames, got {window.shape[0]}")

    valid_center = max(0, min(chunk, features.shape[0] - center_start))
    return window.astype(np.float32), valid_center
