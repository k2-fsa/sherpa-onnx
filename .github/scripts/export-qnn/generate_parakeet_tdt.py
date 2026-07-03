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
    num_seconds: int
    model_name: str


def main():

    #  model_name_list = ["parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3"]
    model_name_list = ["parakeet-tdt-0.6b-v2"]
    num_seconds_list = [3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]

    configs = []

    for name, soc in soc_info_dict.items():
        for num_seconds, model_name in itertools.product(
            num_seconds_list, model_name_list
        ):
            if soc.model.name != "SM8850":
                continue
            configs.append(
                Config(
                    soc=name,
                    soc_id=soc.model.value,
                    arch=soc.info.arch.name,
                    model_name=model_name,
                    num_seconds=num_seconds,
                )
            )

    ans = [asdict(c) for c in configs]

    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    main()
