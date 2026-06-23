#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import json

from device_info import soc_info_dict
from dataclasses import asdict, dataclass
import itertools


@dataclass
class Config:
    soc: str  # SM8850
    soc_id: int  # 87
    arch: str  # v81
    model_name: str


def main():

    # fmt: off
    model_name_list = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "distil-medium.en", "distil-small.en", "medium-aishell"]
    # fmt: on

    configs = []

    for name, soc in soc_info_dict.items():
        if soc.model.name == "SM8350":
            continue
        for model_name in model_name_list:
            configs.append(
                Config(
                    soc=name,
                    soc_id=soc.model.value,
                    arch=soc.info.arch.name,
                    model_name=model_name,
                )
            )

    ans = [asdict(c) for c in configs]

    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    main()
