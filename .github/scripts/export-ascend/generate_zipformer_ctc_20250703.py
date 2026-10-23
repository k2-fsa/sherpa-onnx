#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import itertools
import json
from dataclasses import asdict, dataclass


# Images come from the official quay.io/ascend/cann registry (the gpustack/*
# images on Docker Hub disappeared):
#   https://quay.io/repository/ascend/cann?tab=tags
#   https://github.com/Ascend/cann-container-image/blob/main/supported_tags.md
def get_image(cann: str, soc_version: str):
    cann2image_910 = {
        "7.0": "quay.io/ascend/cann:7.0.1.beta1-910b-ubuntu22.04-py3.8",
        "8.0": "quay.io/ascend/cann:8.0.0-910b-ubuntu22.04-py3.9",
        "8.1": "quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.11",
        "8.2": "quay.io/ascend/cann:8.2.rc1-910b-ubuntu22.04-py3.11",
        "8.3": "quay.io/ascend/cann:8.3.rc2-910b-ubuntu22.04-py3.11",
        "8.5": "quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11",
    }

    cann2image_310 = {
        "7.0": "quay.io/ascend/cann:7.0.1-310p-ubuntu22.04-py3.9",
        "8.0": "quay.io/ascend/cann:8.0.rc1-310p-ubuntu22.04-py3.9",
        "8.1": "quay.io/ascend/cann:8.1.rc1-310p-ubuntu22.04-py3.11",
        "8.2": "quay.io/ascend/cann:8.2.rc1-310p-ubuntu22.04-py3.11",
        "8.3": "quay.io/ascend/cann:8.3.rc2-310p-ubuntu22.04-py3.11",
        "8.5": "quay.io/ascend/cann:8.5.0-310p-ubuntu22.04-py3.11",
    }

    if "910" in soc_version:
        return cann2image_910[cann]
    elif "310" in soc_version:
        return cann2image_310[cann]
    else:
        raise ValueError(f"Unsupported soc_version {soc_version}")


def get_soc_version():
    soc_version = ["910B", "910B2", "910B3", "910B4", "310P3"]
    return soc_version


def get_cann_version():
    cann_version = ["7.0", "8.0", "8.1", "8.2", "8.3", "8.5"]
    return cann_version


@dataclass
class Config:
    # 7.0, 8.0, 8.1, 8.2
    cann: str

    # 910B, 910B2, 910B3, 310P3
    soc_version: str

    num_seconds: str

    image: str = ""

    def __post_init__(self):
        self.image = get_image(self.cann, soc_version=self.soc_version)


def main():
    cann_version = get_cann_version()
    soc_version = get_soc_version()
    input_in_seconds = ["5", "8", "10", "13", "15", "18", "20", "23", "25", "28", "30"]

    configs = [
        Config(cann=cann, soc_version=soc, num_seconds=sec)
        for cann, soc, sec in itertools.product(
            cann_version, soc_version, input_in_seconds
        )
    ]

    ans = [asdict(c) for c in configs]

    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    main()
