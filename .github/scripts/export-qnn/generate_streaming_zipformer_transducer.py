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
    chunk_size: int
    model_name: str


def main():

    model_name_list = ["20230220"]
    chunk_size_list = [32]

    configs = []

    for name, soc in soc_info_dict.items():
        if soc.model.name == "SM8350":
            # SM8350 does not support fp16
            continue
        for chunk_size, model_name in itertools.product(
            chunk_size_list, model_name_list
        ):
            configs.append(
                Config(
                    soc=name,
                    soc_id=soc.model.value,
                    arch=soc.info.arch.name,
                    model_name=model_name,
                    chunk_size=chunk_size,
                )
            )

    ans = [asdict(c) for c in configs]

    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    main()
