#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import List

import jinja2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total",
        type=int,
        default=1,
        help="Number of runners",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the current runner",
    )
    return parser.parse_args()


@dataclass
class Model:
    model_name: str
    short_name: str
    index: int  # index in ModelConfig.kt (SELECTED_MODEL_INDEX)


def get_models() -> List[Model]:
    # The order must match SELECTED_MODEL_INDEX in
    # android/SherpaOnnxForcedAlignment/app/src/main/java/com/k2fsa/sherpa/onnx/fa/ModelConfig.kt
    return [
        Model(model_name="zh-aishell3-pinyin", short_name="zh-aishell3", index=0),
        Model(model_name="zh-biaobei-pinyin", short_name="zh-biaobei", index=1),
        Model(model_name="zh-large-pinyin", short_name="zh-large", index=2),
        Model(model_name="english-mfa-v2.2.1", short_name="en-v2.2.1", index=3),
        Model(model_name="english-mfa-v3.0.0", short_name="en-v3.0.0", index=4),
        Model(model_name="english-mfa-v3.1.0", short_name="en-v3.1.0", index=5),
    ]


def main():
    args = get_args()
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)

    all_model_list = get_models()
    num_models = len(all_model_list)

    num_per_runner = num_models // total
    if num_per_runner <= 0:
        raise ValueError(f"num_models: {num_models}, num_runners: {total}")

    start = index * num_per_runner
    end = start + num_per_runner

    remaining = num_models - args.total * num_per_runner

    print(f"{index}/{total}: {start}-{end}/{num_models}")

    d = dict()
    d["model_list"] = all_model_list[start:end]
    if index < remaining:
        s = args.total * num_per_runner + index
        d["model_list"].append(all_model_list[s])
        print(f"{s}/{num_models}")

    for i, m in enumerate(all_model_list):
        assert m.index == i

    filename_list = ["./build-apk-forced-alignment.sh"]
    for filename in filename_list:
        environment = jinja2.Environment()
        with open(f"{filename}.in") as f:
            s = f.read()
        template = environment.from_string(s)

        s = template.render(**d)
        with open(filename, "w") as f:
            print(s, file=f)


if __name__ == "__main__":
    main()
