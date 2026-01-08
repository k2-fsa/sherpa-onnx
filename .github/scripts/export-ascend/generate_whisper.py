#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import itertools
import json
from dataclasses import asdict, dataclass

from generate_zipformer_ctc_20250703 import get_image, get_soc_version, get_cann_version


@dataclass
class Config:
    # 7.0, 8.0, 8.2
    cann: str

    # 910B, 910B2, 910B3, 310P3
    soc_version: str

    model: str

    image: str = ""

    def __post_init__(self):
        self.image = get_image(self.cann, soc_version=self.soc_version)


def main():
    cann_version = get_cann_version()
    soc_version = get_soc_version()
    model_list = [
        "turbo",
        "distil-medium.en",
        "distil-small.en",
        "tiny.en",
        "base.en",
        "small.en",
        "medium.en",
        "tiny",
        "base",
        "small",
        "medium",
        "medium-aishell",
    ]

    configs = [
        Config(cann=cann, soc_version=soc, model=model)
        for cann, soc, model in itertools.product(cann_version, soc_version, model_list)
    ]

    ans = [asdict(c) for c in configs]

    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    main()
