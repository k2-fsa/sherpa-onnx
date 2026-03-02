#!/usr/bin/env python3
# Copyright    2026  zengyw
# Generate unicode_indexer.bin from unicode_indexer.json.

import json
import sys
from pathlib import Path

import numpy as np


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

    array = np.array(arr, dtype=np.int32)

    with open(bin_path, "wb") as f:
        f.write(array.tobytes())
    print(f"Wrote {len(arr)} int32 -> {bin_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
