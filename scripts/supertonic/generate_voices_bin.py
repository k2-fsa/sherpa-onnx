#!/usr/bin/env python3
# Copyright    2026  zengyw
# Convert Supertonic voice style JSON files to binary format
import json
import struct
from pathlib import Path


def flatten_3d_array(data):
    result = []
    for batch in data:
        for row in batch:
            result.extend(row)
    return result


def json_to_binary(json_path, output_path):
    """Convert a Supertonic voice style JSON file to binary format.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    style_ttl = data["style_ttl"]
    ttl_dims = style_ttl["dims"]
    ttl_data_nested = style_ttl["data"]
    ttl_data_flat = flatten_3d_array(ttl_data_nested)

    style_dp = data["style_dp"]
    dp_dims = style_dp["dims"]
    dp_data_nested = style_dp["data"]
    dp_data_flat = flatten_3d_array(dp_data_nested)

    ttl_expected_size = 1
    for d in ttl_dims:
        ttl_expected_size *= d
    if len(ttl_data_flat) != ttl_expected_size:
        raise ValueError(
            f"ttl_dims product ({ttl_expected_size}) != ttl_data size "
            f"({len(ttl_data_flat)})"
        )

    dp_expected_size = 1
    for d in dp_dims:
        dp_expected_size *= d
    if len(dp_data_flat) != dp_expected_size:
        raise ValueError(
            f"dp_dims product ({dp_expected_size}) != dp_data size "
            f"({len(dp_data_flat)})"
        )

    with open(output_path, "wb") as f:
        for d in ttl_dims:
            f.write(struct.pack("<q", d))
        for d in dp_dims:
            f.write(struct.pack("<q", d))
        for val in ttl_data_flat:
            f.write(struct.pack("<f", val))
        for val in dp_data_flat:
            f.write(struct.pack("<f", val))


def main():
    """Convert all voice style JSON files to binary format."""
    voice_styles_dir = Path(__file__).parent / "assets" / "voice_styles"
    
    if not voice_styles_dir.exists():
        print(f"Error: {voice_styles_dir} does not exist")
        return
    
    json_files = sorted(voice_styles_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {voice_styles_dir}")
        return
    
    print(f"Found {len(json_files)} voice style JSON files")
    
    for json_file in json_files:
        # Output binary file with .bin extension
        bin_file = json_file.with_suffix('.bin')
        
        if bin_file.exists():
            print(f"{bin_file.name} exists - skip")
            continue
        
        try:
            print(f"Converting {json_file.name} -> {bin_file.name}...")
            json_to_binary(json_file, bin_file)
            print("Success !")
        except Exception as e:
            print(f"Error converting {json_file.name}: {e}")


if __name__ == "__main__":
    main()
