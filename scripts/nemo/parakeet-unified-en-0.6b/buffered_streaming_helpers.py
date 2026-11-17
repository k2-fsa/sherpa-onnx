#!/usr/bin/env python3
# Copyright      2026  Milan Leonard

import numpy as np


def normalize_per_feature(features: np.ndarray) -> np.ndarray:
    reference = features[:1]
    mean = reference + (features - reference).mean(axis=0, keepdims=True)
    variance = ((features - mean) ** 2).sum(axis=0, keepdims=True)
    variance /= max(len(features) - 1, 1)
    std = np.sqrt(variance) + 1e-5
    return ((features - mean) / std).astype(np.float32)


def slice_feature_buffer(
    features: np.ndarray,
    center_start: int,
    left: int,
    chunk: int,
    right: int,
):
    remaining = max(0, features.shape[0] - center_start)
    valid_center = remaining if remaining <= chunk + right else chunk
    right_frames = min(right, remaining - valid_center)
    max_left = left + chunk + right - valid_center if right_frames == 0 else left
    start = max(0, center_start - max_left)
    end = center_start + valid_center + right_frames
    # Missing context must not contribute zeros to per-feature normalization.
    window = features[start:end]
    return window.astype(np.float32), valid_center
