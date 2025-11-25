#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import itertools
import json
from dataclasses import asdict, dataclass

from generate_zipformer_ctc_20250703 import get_image


@dataclass
class Config:
    # 7.0, 8.0, 8.2
    cann: str

    # 910B, 910B2, 910B3, 310P3
    soc_version: str

    # FunASR, WSYue-ASR
    framework: str

    image: str = ""

    def __post_init__(self):
        self.image = get_image(self.cann, soc_version=self.soc_version)


def main():
    cann_version = ["7.0", "8.0", "8.2"]
    soc_version = ["910B", "910B2", "910B3", "310P3"]
    framework_list = ["FunASR", "WSYue-ASR"]

    configs = [
        Config(cann=cann, soc_version=soc, framework=framework)
        for cann, soc, framework in itertools.product(
            cann_version, soc_version, framework_list
        )
    ]

    ans = [asdict(c) for c in configs]

    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    main()
