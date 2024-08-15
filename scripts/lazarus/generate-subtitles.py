#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import List, Optional

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
    lang: str
    short_name: str = ""
    cmd: str = ""


def get_models():
    models = [
        Model(
            model_name="sherpa-onnx-whisper-tiny.en",
            lang="en",
            short_name="whisper_tiny.en",
            cmd="""
            pushd $model_name
            rm -fv tiny.en-encoder.onnx
            rm -fv tiny.en-decoder.onnx

            mv -v tiny.en-encoder.int8.onnx whisper-encoder.onnx
            mv -v tiny.en-decoder.int8.onnx whisper-decoder.onnx
            mv -v tiny.en-tokens.txt tokens.txt

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            lang="zh_en_ko_ja_yue",
            short_name="sense_voice",
            cmd="""
            pushd $model_name
            rm -fv model.onnx
            mv -v model.int8.onnx sense-voice.onnx
            popd
            """,
        ),
    ]
    return models


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

    filename_list = [
        "./build-generate-subtitles.sh",
    ]
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
