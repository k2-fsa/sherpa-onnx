#!/usr/bin/env python3

from dataclasses import dataclass

import jinja2
from typing import List


@dataclass
class TtsModel:
    model_dir: str
    model_name: str
    lang: str  # en, zh, fr, de, etc.


def get_all_models() -> List[TtsModel]:
    return [
        TtsModel(model_dir="vits-ljs", model_name="vits-ljs.onnx", lang="en"),
        TtsModel(model_dir="vits-vctk", model_name="vits-vctk.onnx", lang="en"),
        TtsModel(
            model_dir="vits-zh-aishell3", model_name="vits-aishell3.onnx", lang="en"
        ),
        TtsModel(
            model_dir="vits-piper-en_US-lessac-medium",
            model_name="en_US-lessac-medium.onnx",
            lang="en",
        ),
    ]


def main():
    environment = jinja2.Environment()
    with open("./build-apk-tts.sh.in") as f:
        s = f.read()
    template = environment.from_string(s)
    d = dict()
    d["tts_model_list"] = get_all_models()
    s = template.render(**d)
    with open("./build-apk-tts.sh", "w") as f:
        print(s, file=f)


if __name__ == "__main__":
    main()
