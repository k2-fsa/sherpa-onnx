#!/usr/bin/env python3
# Copyright    2026  zengyw
# Generate tts.bin from tts.json (fixed 4 x int32: sample_rate, base_chunk_size,
# chunk_compress_factor, latent_dim) for runtime.
# Usage: python3 generate_tts_bin.py [json_path] [bin_path]

import json
import struct
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    default_json = script_dir.parent.parent / "assets" / "onnx" / "tts.json"
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    bin_path = Path(sys.argv[2]) if len(sys.argv) > 2 else json_path.with_suffix(
        ".bin"
    )

    if not json_path.exists():
        print(f"Error: {json_path} does not exist")
        return 1
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    if not isinstance(j, dict):
        print(f"Error: JSON must be an object, got {type(j)}")
        return 1
    for key in ("ae", "ttl"):
        if key not in j or not isinstance(j[key], dict):
            print(f"Error: JSON must contain '{key}' object")
            return 1
    ae = j["ae"]
    ttl = j["ttl"]
    for name, val in [
        ("ae.sample_rate", ae.get("sample_rate")),
        ("ae.base_chunk_size", ae.get("base_chunk_size")),
        ("ttl.chunk_compress_factor", ttl.get("chunk_compress_factor")),
        ("ttl.latent_dim", ttl.get("latent_dim")),
    ]:
        if val is None:
            print(f"Error: missing {name}")
            return 1
        if not isinstance(val, int) or val < -(2**31) or val > 2**31 - 1:
            print(f"Error: {name} must be int32, got {val}")
            return 1
    sample_rate = int(ae["sample_rate"])
    base_chunk_size = int(ae["base_chunk_size"])
    chunk_compress_factor = int(ttl["chunk_compress_factor"])
    latent_dim = int(ttl["latent_dim"])
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<4i", sample_rate, base_chunk_size,
                            chunk_compress_factor, latent_dim))
    print(f"Wrote tts.bin (sample_rate=%d, base_chunk_size=%d, "
          "chunk_compress_factor=%d, latent_dim=%d) -> %s" % (
              sample_rate, base_chunk_size, chunk_compress_factor, latent_dim,
              bin_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
