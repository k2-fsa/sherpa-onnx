#!/usr/bin/env python3
# Copyright    2026  zengyw
# Merge Supertonic voice style JSONs from a directory into one voice.bin
# (multi-speaker; use --sid 0..N-1 at runtime).
# Usage: python3 generate_voices_bin.py [input_dir] [output_bin]

import json
import struct
import sys
from pathlib import Path


def flatten_3d_array(data):
    result = []
    for batch in data:
        for row in batch:
            result.extend(row)
    return result


def load_one_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    style_ttl = data["style_ttl"]
    ttl_dims = style_ttl["dims"]
    ttl_data_flat = flatten_3d_array(style_ttl["data"])
    style_dp = data["style_dp"]
    dp_dims = style_dp["dims"]
    dp_data_flat = flatten_3d_array(style_dp["data"])
    ttl_size = 1
    for d in ttl_dims:
        ttl_size *= d
    if len(ttl_data_flat) != ttl_size:
        raise ValueError(
            f"ttl_dims product ({ttl_size}) != ttl_data size ({len(ttl_data_flat)})"
        )
    dp_size = 1
    for d in dp_dims:
        dp_size *= d
    if len(dp_data_flat) != dp_size:
        raise ValueError(
            f"dp_dims product ({dp_size}) != dp_data size ({len(dp_data_flat)})"
        )
    return ttl_dims, ttl_data_flat, dp_dims, dp_data_flat


def merge_jsons_to_binary(json_paths, output_path):
    """Merge N JSONs into one voice.bin (shape [N, d1, d2])."""
    if not json_paths:
        raise ValueError("No JSON paths given")
    ttl_dims_list = []
    ttl_flat_list = []
    dp_dims_list = []
    dp_flat_list = []
    for p in json_paths:
        ttl_dims, ttl_flat, dp_dims, dp_flat = load_one_json(p)
        if len(ttl_dims) != 3 or ttl_dims[0] != 1:
            raise ValueError(
                f"{p}: expected ttl dims [1, d1, d2], got {ttl_dims}"
            )
        if len(dp_dims) != 3 or dp_dims[0] != 1:
            raise ValueError(
                f"{p}: expected dp dims [1, d1, d2], got {dp_dims}"
            )
        ttl_dims_list.append(ttl_dims)
        ttl_flat_list.append(ttl_flat)
        dp_dims_list.append(dp_dims)
        dp_flat_list.append(dp_flat)
    ref_ttl = ttl_dims_list[0]
    ref_dp = dp_dims_list[0]
    for i, (ttl, dp) in enumerate(zip(ttl_dims_list, dp_dims_list)):
        if ttl[1:] != ref_ttl[1:] or dp[1:] != ref_dp[1:]:
            raise ValueError(
                f"File {json_paths[i]} has dims ttl{ttl} dp{dp}; "
                f"expected ttl[1:]={ref_ttl[1:]}, dp[1:]={ref_dp[1:]}"
            )
    n = len(json_paths)
    out_ttl_dims = [n, ref_ttl[1], ref_ttl[2]]
    out_dp_dims = [n, ref_dp[1], ref_dp[2]]
    with open(output_path, "wb") as f:
        for d in out_ttl_dims:
            f.write(struct.pack("<q", d))
        for d in out_dp_dims:
            f.write(struct.pack("<q", d))
        for flat in ttl_flat_list:
            for val in flat:
                f.write(struct.pack("<f", val))
        for flat in dp_flat_list:
            for val in flat:
                f.write(struct.pack("<f", val))
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
