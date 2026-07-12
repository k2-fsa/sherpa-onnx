#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import json
import sys

from device_info import soc_info_dict
from dataclasses import asdict, dataclass
import itertools


model_name_list = [
    #  "moonshine-base-zh",
    #  "moonshine-base-ja",
    #  "moonshine-base-uk",
    #  "moonshine-base-vi",
    #  "moonshine-base-ar",
    #  "moonshine-base-ko",
    #  "moonshine-tiny-ja",
    # 210
    "moonshine-tiny-ar",
    "moonshine-tiny-ko",
    "moonshine-tiny-zh",
    "moonshine-tiny-vi",
    "moonshine-tiny-uk",
    "moonshine-base",
    "moonshine-tiny",
]

num_seconds_list = [5, 8, 10]


@dataclass
class Config:
    soc: str  # SM8850
    soc_id: int  # 87
    arch: str  # v81
    num_seconds: int
    model_name: str


def main():
    configs = []

    for name, soc in soc_info_dict.items():
        for num_seconds, model_name in itertools.product(
            num_seconds_list, model_name_list
        ):
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
    if len(sys.argv) > 1 and sys.argv[1] == "--model-names":
        print(
            json.dumps({"model_name": model_name_list, "num_seconds": num_seconds_list})
        )
    else:
        main()
