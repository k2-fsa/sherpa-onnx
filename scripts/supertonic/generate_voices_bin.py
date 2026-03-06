#!/usr/bin/env python3
# Copyright    2026  zengyw
# Merge Supertonic voice style JSONs from a directory into one voice.bin

import json
import sys
from pathlib import Path

import numpy as np


def load_one_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "style_ttl" not in data:
        raise ValueError(f"{json_path}: missing key 'style_ttl'")
    if "style_dp" not in data:
        raise ValueError(f"{json_path}: missing key 'style_dp'")

    style_ttl = data["style_ttl"]
    if "dims" not in style_ttl or "data" not in style_ttl:
        raise ValueError(f"{json_path}: 'style_ttl' must contain keys 'dims' and 'data'")
    ttl_dims = tuple(int(x) for x in style_ttl["dims"])
    ttl_arr = np.asarray(style_ttl["data"], dtype=np.float32)

    ttl_size = int(np.prod(ttl_dims)) if len(ttl_dims) > 0 else 0
    if ttl_arr.size != ttl_size:
        raise ValueError(
            f"{json_path}: ttl size {ttl_arr.size} != prod(ttl_dims) {ttl_size} (ttl_dims={ttl_dims})"
        )
    ttl_arr = ttl_arr.reshape(ttl_dims)
    if not np.all(np.isfinite(ttl_arr)):
        raise ValueError(f"{json_path}: ttl contains NaN/Inf")

    style_dp = data["style_dp"]
    if "dims" not in style_dp or "data" not in style_dp:
        raise ValueError(f"{json_path}: 'style_dp' must contain keys 'dims' and 'data'")
    dp_dims = tuple(int(x) for x in style_dp["dims"])
    dp_arr = np.asarray(style_dp["data"], dtype=np.float32)

    dp_size = int(np.prod(dp_dims)) if len(dp_dims) > 0 else 0
    if dp_arr.size != dp_size:
        raise ValueError(
            f"{json_path}: dp size {dp_arr.size} != prod(dp_dims) {dp_size} (dp_dims={dp_dims})"
        )
    dp_arr = dp_arr.reshape(dp_dims)
    if not np.all(np.isfinite(dp_arr)):
        raise ValueError(f"{json_path}: dp contains NaN/Inf")
    return ttl_dims, ttl_arr, dp_dims, dp_arr


def merge_jsons_to_binary(json_paths, output_path):
    if not json_paths:
        raise ValueError("No JSON paths given")
    ttl_arrays = []
    dp_arrays = []
    ref_ttl = ref_dp = None
    for p in json_paths:
        ttl_dims, ttl_arr, dp_dims, dp_arr = load_one_json(p)
        if len(ttl_dims) != 3 or ttl_dims[0] != 1:
            raise ValueError(
                f"{p}: expected ttl dims [1, d1, d2], got {ttl_dims}"
            )
        if len(dp_dims) != 3 or dp_dims[0] != 1:
            raise ValueError(
                f"{p}: expected dp dims [1, d1, d2], got {dp_dims}"
            )
        if ref_ttl is None:
            ref_ttl, ref_dp = ttl_dims, dp_dims
        elif ttl_dims[1:] != ref_ttl[1:] or dp_dims[1:] != ref_dp[1:]:
            raise ValueError(
                f"File {p} has dims ttl{ttl_dims} dp{dp_dims}; "
                f"expected ttl[1:]={ref_ttl[1:]}, dp[1:]={ref_dp[1:]}"
            )
        ttl_arrays.append(ttl_arr)
        dp_arrays.append(dp_arr)

    n = len(json_paths)
    ttl_stack = np.concatenate(ttl_arrays, axis=0)
    dp_stack = np.concatenate(dp_arrays, axis=0)
    out_ttl_dims = np.array([n, ref_ttl[1], ref_ttl[2]], dtype=np.int64)
    out_dp_dims = np.array([n, ref_dp[1], ref_dp[2]], dtype=np.int64)

    with open(output_path, "wb") as f:
        f.write(out_ttl_dims.tobytes())
        f.write(out_dp_dims.tobytes())
        f.write(ttl_stack.ravel().tobytes())
        f.write(dp_stack.ravel().tobytes())
    print(f"Merged {n} voice(s) -> {output_path} (sid 0..{n - 1})")


def main():
    script_dir = Path(__file__).parent
    default_input = script_dir / "assets" / "voice_styles"
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_input
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_dir / "voice.bin"

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input dir does not exist or not a directory: {input_dir}")
        return 1
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 1

    try:
        merge_jsons_to_binary([str(p) for p in json_files], str(output_path))
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
