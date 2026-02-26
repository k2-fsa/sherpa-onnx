#!/usr/bin/env python3
# Copyright    2026  zengyw
# Generate unicode_indexer.bin from unicode_indexer.json (raw int32 array) for
# runtime. Usage: python3 generate_indexer_bin.py [json_path] [bin_path]

import json
import struct
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    default_json = script_dir.parent.parent / "assets" / "onnx" / "unicode_indexer.json"
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    bin_path = Path(sys.argv[2]) if len(sys.argv) > 2 else json_path.with_suffix(".bin")

    if not json_path.exists():
        print(f"Error: {json_path} does not exist")
        return 1
    with open(json_path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    if not isinstance(arr, list):
        print(f"Error: JSON must be an array of integers, got {type(arr)}")
        return 1
    with open(bin_path, "wb") as f:
        for v in arr:
            vi = int(v)
            if vi < -(2**31) or vi > 2**31 - 1:
                print(f"Error: value {vi} out of int32 range")
                return 1
            f.write(struct.pack("<i", vi))
    print(f"Wrote {len(arr)} int32 -> {bin_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
