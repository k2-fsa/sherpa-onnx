#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import json
import math
import sys

from device_info import soc_info_dict
from dataclasses import asdict, dataclass
import itertools

MAX_JOBS = 220

model_name_list = [
    "moonshine-base-zh",
    "moonshine-base-ja",
    "moonshine-base-uk",
    "moonshine-base-vi",
    "moonshine-base-ar",
    "moonshine-base-ko",
    "moonshine-tiny-ja",
    "moonshine-tiny-ar",
    "moonshine-tiny-ko",
    "moonshine-tiny-zh",
    "moonshine-tiny-vi",
    "moonshine-tiny-uk",
    "moonshine-base",
    "moonshine-tiny",
]

num_seconds_list = [5, 8, 10]

num_socs = len(soc_info_dict)
total_combos = len(model_name_list) * len(num_seconds_list)
total_qnn_jobs = total_combos * num_socs
num_batches = math.ceil(total_qnn_jobs / MAX_JOBS)


@dataclass
class Config:
    soc: str
    soc_id: int
    arch: str
    num_seconds: int
    model_name: str


def make_configs(models):
    configs = []
    for name, soc in soc_info_dict.items():
        for num_seconds, model_name in itertools.product(num_seconds_list, models):
            configs.append(
                Config(
                    soc=name,
                    soc_id=soc.model.value,
                    arch=soc.info.arch.name,
                    model_name=model_name,
                    num_seconds=num_seconds,
                )
            )
    return configs


def main():
    configs = make_configs(model_name_list)
    ans = [asdict(c) for c in configs]
    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--model-names":
        print(
            json.dumps({"model_name": model_name_list, "num_seconds": num_seconds_list})
        )

    elif len(sys.argv) > 1 and sys.argv[1] == "--num-batches":
        print(num_batches)

    elif len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_idx = int(sys.argv[2])
        total_batches = int(sys.argv[3])
        batch_size = math.ceil(len(model_name_list) / total_batches)
        start = batch_idx * batch_size
        end = min(start + batch_size, len(model_name_list))
        batch_models = model_name_list[start:end]

        configs = make_configs(batch_models)
        ans = [asdict(c) for c in configs]
        print(json.dumps({"include": ans}))

    else:
        main()
