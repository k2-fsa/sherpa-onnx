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

    @property
    def dst_suffix(self) -> str:
        return f"cuda-{self.cuda_version}.x-cudnn-9.x-onnxruntime{self.onnxruntime_version}-linux-x64-gpu"


# To add a new onnxruntime version, just add entries here.
onnxruntime_configs = [
    OnnxruntimeConfig("1.27.1", "12"),
    OnnxruntimeConfig("1.27.1", "13"),
]

build_types = ["Release"]


def main():
    entries = []
    for build_type, cfg in itertools.product(build_types, onnxruntime_configs):
        d = asdict(cfg)
        d["os"] = "ubuntu-latest"
        d["build_type"] = build_type
        d["onnxruntime_url_suffix"] = cfg.onnxruntime_url_suffix
        d["onnxruntime_dir_name"] = cfg.onnxruntime_dir_name
        d["cuda_version_tag"] = cfg.cuda_version_tag
        d["dst_suffix"] = cfg.dst_suffix
        entries.append(d)

    print(json.dumps({"include": entries}))


if __name__ == "__main__":
    main()
