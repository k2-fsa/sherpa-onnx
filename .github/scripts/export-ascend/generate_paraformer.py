#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import itertools
import json
from dataclasses import asdict, dataclass

from generate_zipformer_ctc_20250703 import get_cann_version, get_image, get_soc_version


@dataclass
class Config:
    # 7.0, 8.0, 8.1, 8.2
    cann: str

    # 910B, 910B2, 910B3, 310P3
    soc_version: str

    # FunASR, WSChuan-ASR
    framework: str

    image: str = ""

    def __post_init__(self):
        self.image = get_image(self.cann, soc_version=self.soc_version)


def main():
    cann_version = get_cann_version()
    soc_version = get_soc_version()
    framework_list = ["FunASR", "WSChuan-ASR"]

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
