#!/usr/bin/env python3

from dataclasses import dataclass

import jinja2
from typing import List
import argparse


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
class TtsModel:
    model_dir: str
    model_name: str
    lang: str  # en, zh, fr, de, etc.


def get_all_models() -> List[TtsModel]:
    return [
        TtsModel(
            model_dir="vits-zh-aishell3", model_name="vits-aishell3.onnx", lang="zh"
        ),
        # English (US)
        # fmt: off
        TtsModel(model_dir="vits-vctk", model_name="vits-vctk.onnx", lang="en"),
        TtsModel(model_dir="vits-ljs", model_name="vits-ljs.onnx", lang="en"),
        TtsModel(model_dir="vits-piper-en_US-amy-low", model_name="en_US-amy-low.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-amy-medium", model_name="en_US-amy-medium.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-arctic-medium", model_name="en_US-arctic-medium.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-danny-low", model_name="en_US-danny-low.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-hfc_male-medium", model_name="en_US-hfc_male-medium.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-joe-medium", model_name="en_US-joe-medium.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-kathleen-low", model_name="en_US-kathleen-low.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-kusal-medium", model_name="en_US-kusal-medium.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-l2arctic-medium", model_name="en_US-l2arctic-medium.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-lessac-low", model_name="en_US-lessac-low.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-lessac-medium", model_name="en_US-lessac-medium.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-lessac-high", model_name="en_US-lessac-high.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-libritts-high", model_name="en_US-libritts-high.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-libritts_r-medium", model_name="en_US-libritts_r-medium.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-ryan-low", model_name="en_US-ryan-low.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-ryan-medium", model_name="en_US-ryan-medium.onnx", lang="en",),
        TtsModel(model_dir="vits-piper-en_US-ryan-high", model_name="en_US-ryan-high.onnx", lang="en",),
        # English (GB)
        TtsModel(model_dir="vits-piper-en_GB-alan-low", model_name="en_GB-alan-low.onnx",lang="en",),
        TtsModel(model_dir="vits-piper-en_GB-alan-medium", model_name="en_GB-alan-medium.onnx",lang="en",),
        TtsModel(model_dir="vits-piper-en_GB-alba-medium", model_name="en_GB-alba-medium.onnx",lang="en",),
        TtsModel(model_dir="vits-piper-en_GB-jenny_dioco-medium", model_name="en_GB-jenny_dioco-medium.onnx",lang="en",),
        TtsModel(model_dir="vits-piper-en_GB-northern_english_male-medium", model_name="en_GB-northern_english_male-medium.onnx",lang="en",),
        TtsModel(model_dir="vits-piper-en_GB-semaine-medium", model_name="en_GB-semaine-medium.onnx",lang="en",),
        TtsModel(model_dir="vits-piper-en_GB-southern_english_female-low", model_name="en_GB-southern_english_female-low.onnx",lang="en",),
        TtsModel(model_dir="vits-piper-en_GB-vctk-medium", model_name="en_GB-vctk-medium.onnx",lang="en",),
        # German (DE)
        TtsModel(model_dir="vits-piper-de_DE-eva_k-x_low", model_name="de_DE-eva_k-x_low.onnx",lang="de",),
        TtsModel(model_dir="vits-piper-de_DE-karlsson-low", model_name="de_DE-karlsson-low.onnx",lang="de",),
        TtsModel(model_dir="vits-piper-de_DE-kerstin-low", model_name="de_DE-kerstin-low.onnx",lang="de",),
        TtsModel(model_dir="vits-piper-de_DE-pavoque-low", model_name="de_DE-pavoque-low.onnx",lang="de",),
        TtsModel(model_dir="vits-piper-de_DE-ramona-low", model_name="de_DE-ramona-low.onnx",lang="de",),
        TtsModel(model_dir="vits-piper-de_DE-thorsten-low", model_name="de_DE-thorsten-low.onnx",lang="de",),
        TtsModel(model_dir="vits-piper-de_DE-thorsten-medium", model_name="de_DE-thorsten-medium.onnx",lang="de",),
        TtsModel(model_dir="vits-piper-de_DE-thorsten-high", model_name="de_DE-thorsten-high.onnx",lang="de",),
        TtsModel(model_dir="vits-piper-de_DE-thorsten_emotional-medium", model_name="de_DE-thorsten_emotional-medium.onnx",lang="de",),
        # French (FR)
        TtsModel(model_dir="vits-piper-fr_FR-upmc-medium", model_name="fr_FR-upmc-medium.onnx",lang="fr",),
        TtsModel(model_dir="vits-piper-fr_FR-siwis-low", model_name="fr_FR-siwis-low.onnx",lang="fr",),
        TtsModel(model_dir="vits-piper-fr_FR-siwis-medium", model_name="fr_FR-siwis-medium.onnx",lang="fr",),

        # Spanish (ES)
        TtsModel(model_dir="vits-piper-es_ES-carlfm-x_low", model_name="es_ES-carlfm-x_low.onnx",lang="es",),
        TtsModel(model_dir="vits-piper-es_ES-davefx-medium", model_name="es_ES-davefx-medium.onnx",lang="es",),
        TtsModel(model_dir="vits-piper-es_ES-mls_10246-low", model_name="es_ES-mls_10246-low.onnx",lang="es",),
        TtsModel(model_dir="vits-piper-es_ES-mls_9972-low", model_name="es_ES-mls_9972-low.onnx",lang="es",),
        TtsModel(model_dir="vits-piper-es_ES-sharvard-medium", model_name="es_ES-sharvard-medium.onnx",lang="es",),

        # Spanish (MX)
        TtsModel(model_dir="vits-piper-es_MX-ald-medium", model_name="es_MX-ald-medium.onnx",lang="es",),
        # fmt: on
    ]


def main():
    args = get_args()
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)
    environment = jinja2.Environment()
    with open("./build-apk-tts.sh.in") as f:
        s = f.read()
    template = environment.from_string(s)
    d = dict()
    all_model_list = get_all_models()
    num_models = len(all_model_list)

    num_per_runner = num_models // total
    if num_per_runner <= 0:
        raise ValueError(f"num_models: {num_models}, num_runners: {total}")

    start = index * num_per_runner
    end = start + num_per_runner
    if index == args.total - 1:
        end = num_models

    print(f"{index}/{total}: {start}-{end}/{num_models}")
    d["tts_model_list"] = all_model_list[start:end]
    s = template.render(**d)
    with open("./build-apk-tts.sh", "w") as f:
        print(s, file=f)


if __name__ == "__main__":
    main()
