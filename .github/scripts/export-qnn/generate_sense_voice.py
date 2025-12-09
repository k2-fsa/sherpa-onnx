#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import json

from device_info import soc_info_dict
from dataclasses import asdict, dataclass
import itertools


@dataclass
class Config:
    soc: str  # SM8850
    soc_id: int  # 87
    arch: str  # v81
    input_in_seconds: str
    framework: str


def main():

    input_in_seconds = ["5", "8", "10", "13", "15", "18", "20", "23", "25", "28", "30"]
    input_in_seconds = ["5", "10"]
    framework_list = ["FunASR", "WSYue-ASR"]
    framework_list = ["FunASR"]

    configs = []

    for name, soc in soc_info_dict.items():
        if name != "SM8850":
            continue

        for num_seconds, framework in itertools.product(
            input_in_seconds, framework_list
        ):
            configs.append(
                Config(
                    soc=name,
                    soc_id=soc.model.value,
                    arch=soc.info.arch.name,
                    input_in_seconds=num_seconds,
                    framework=framework,
                )
            )

    ans = [asdict(c) for c in configs]

    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    main()
