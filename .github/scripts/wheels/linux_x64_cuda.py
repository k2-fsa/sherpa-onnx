#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import itertools
import json
from dataclasses import asdict, dataclass


@dataclass
class OnnxruntimeConfig:
    onnxruntime_version: str
    cuda_version: str  # "12" or "13"

    @property
    def onnxruntime_url_suffix(self) -> str:
        return f"gpu_cuda{self.cuda_version}-{self.onnxruntime_version}-patched"

    @property
    def onnxruntime_dir_name(self) -> str:
        return f"onnxruntime-linux-x64-{self.onnxruntime_url_suffix}"

    @property
    def cuda_version_tag(self) -> str:
        return f"{self.cuda_version}.cudnn9.onnxruntime{self.onnxruntime_version}"


# To add a new onnxruntime version, just add entries here.
onnxruntime_configs = [
    OnnxruntimeConfig("1.27.1", "12"),
    OnnxruntimeConfig("1.27.1", "13"),
]

python_versions = ["3.7", "3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]


def main():
    entries = []
    for py, cfg in itertools.product(python_versions, onnxruntime_configs):
        d = asdict(cfg)
        d["os"] = "ubuntu-22.04"
        d["python-version"] = py
        d["onnxruntime_url_suffix"] = cfg.onnxruntime_url_suffix
        d["onnxruntime_dir_name"] = cfg.onnxruntime_dir_name
        d["cuda_version_tag"] = cfg.cuda_version_tag
        entries.append(d)

    print(json.dumps({"include": entries}))


if __name__ == "__main__":
    main()
