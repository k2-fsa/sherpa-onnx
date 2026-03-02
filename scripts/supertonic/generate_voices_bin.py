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
    style_ttl = data["style_ttl"]
    ttl_dims = tuple(style_ttl["dims"])
    ttl_arr = np.array(style_ttl["data"], dtype=np.float32)

    if ttl_arr.shape != ttl_dims:
        raise ValueError(
            f"ttl shape {ttl_arr.shape} != ttl_dims {ttl_dims}"
        )
    style_dp = data["style_dp"]
    dp_dims = tuple(style_dp["dims"])
    dp_arr = np.array(style_dp["data"], dtype=np.float32)

    if dp_arr.shape != dp_dims:
        raise ValueError(
            f"dp shape {dp_arr.shape} != dp_dims {dp_dims}"
        )
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
