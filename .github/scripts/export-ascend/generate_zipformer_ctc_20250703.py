#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import itertools
import json
from dataclasses import asdict, dataclass


def get_image(cann: str, soc_version: str):
    cann2image_910 = {
        "7.0": "quay.io/ascend/cann:7.0.1.beta1-910b-ubuntu22.04-py3.8",
        "8.0": "gpustack/ascendai-cann:8.0.RC3-910b-ubuntu20.04-py3.9",
        "8.2": "gpustack/devel-ascendai-cann:8.2.rc1-910b-ubuntu20.04-v2",
    }

    cann2image_310 = {
        "7.0": "quay.io/ascend/cann:7.0.1-310p-ubuntu22.04-py3.9",
        "8.0": "gpustack/devel-ascendai-cann:8.0.rc3.beta1-310p-ubuntu20.04-v2",
        "8.2": "gpustack/devel-ascendai-cann:8.2.rc1-310p-ubuntu20.04-v2",
    }
    if "910" in soc_version:
        return cann2image_910[cann]
    elif "310" in soc_version:
        return cann2image_310[cann]
    else:
        raise ValueError(f"Unsupported soc_version {soc_version}")


@dataclass
class Config:
    # 7.0, 8.0, 8.2
    cann: str

    # 910B, 910B2, 910B3, 310P3
    soc_version: str

    num_seconds: int

    image: str = ""

    def __post_init__(self):
        self.image = get_image(self.cann, soc_version=self.soc_version)


def main():
    cann_version = ["7.0", "8.0", "8.2"]
    soc_version = ["910B", "910B2", "910B3", "310P3"]
    input_in_seconds = ["5", "8", "10", "13", "15", "18", "20", "23", "25", "28", "30"]

    configs = []
    for cann, soc, sec in itertools.product(
        cann_version, soc_version, input_in_seconds
    ):
        c = Config(cann=cann, soc_version=soc, num_seconds=sec)
        configs.append(c)

    ans = []
    for c in configs:
        ans.append(asdict(c))

    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    main()
