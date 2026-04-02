#!/usr/bin/env python3

import argparse
from dataclasses import dataclass

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
    hf: str  # huggingface space name
    ms: str  # modelscope space name
    cmd: str = ""


def get_models():
    models = [
        Model(
            model_name="vits-piper-de_DE-thorsten_emotional-medium",
            hf="k2-fsa/web-assembly-tts-sherpa-onnx-de",
            ms="k2-fsa/web-assembly-tts-sherpa-onnx-de",
            cmd="""
            pushd $model_name

            mv -v *.onnx ../
            mv -v tokens.txt ../
            mv -v espeak-ng-data ../
            popd


            git checkout .

            rm -rf $model_name
            git diff
            """,
        ),
        Model(
            model_name="vits-piper-en_US-libritts_r-medium",
            hf="k2-fsa/web-assembly-tts-sherpa-onnx-en",
            ms="k2-fsa/web-assembly-tts-sherpa-onnx-en",
            cmd="""
            pushd $model_name

            mv -v *.onnx ../
            mv -v tokens.txt ../
            mv -v espeak-ng-data ../
            popd


            git checkout .

            rm -rf $model_name
            git diff
            """,
        ),
        Model(
            model_name="matcha-icefall-zh-en",
            hf="k2-fsa/web-assembly-zh-en-tts-matcha",
            ms="csukuangfj/web-assembly-zh-en-tts-matcha",
            cmd="""
            pushd $model_name

            mv -v *.fst ../
            mv -v *.onnx ../
            mv -v tokens.txt ../
            mv -v lexicon.txt ../
            mv -v espeak-ng-data ../
            popd

            curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx

            git checkout .
            sed -i.bak 's/let modelType = 0/let modelType = 1/g' ../sherpa-onnx-tts.js

            rm -rf $model_name
            git diff
            """,
        ),
        Model(
            model_name="matcha-icefall-zh-baker",
            hf="k2-fsa/web-assembly-zh-tts-matcha",
            ms="csukuangfj/web-assembly-zh-tts-matcha",
            cmd="""
            pushd $model_name

            mv -v *.fst ../
            mv -v *.onnx ../
            mv -v tokens.txt ../
            mv -v lexicon.txt ../
            popd

            curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx


            git checkout .
            sed -i.bak 's/let modelType = 0/let modelType = 2/g' ../sherpa-onnx-tts.js

            rm -rf $model_name
            git diff
            """,
        ),
        Model(
            model_name="matcha-icefall-en_US-ljspeech",
            hf="k2-fsa/web-assembly-en-tts-matcha",
            ms="csukuangfj/web-assembly-en-tts-matcha",
            cmd="""
            pushd $model_name

            mv -v *.onnx ../
            mv -v tokens.txt ../
            mv -v espeak-ng-data ../
            popd

            curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx


            git checkout .
            sed -i.bak 's/let modelType = 0/let modelType = 3/g' ../sherpa-onnx-tts.js

             rm -rf $model_name
             git diff
             """,
        ),
        Model(
            model_name="sherpa-onnx-zipvoice-distill-int8-zh-en-emilia",
            hf="k2-fsa/web-assembly-zh-en-tts-zipvoice",
            ms="csukuangfj/web-assembly-zh-en-tts-zipvoice",
            cmd="""
            pushd $model_name

            mv -v encoder.int8.onnx ../
            mv -v decoder.int8.onnx ../
            mv -v tokens.txt ../
            mv -v lexicon.txt ../
            mv -v espeak-ng-data ../
            popd

            curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

            git checkout .
            sed -i.bak 's/let modelType = 0/let modelType = 4/g' ../sherpa-onnx-tts.js
            rm -rf $model_name
            git diff
            """,
        ),
        Model(
            model_name="sherpa-onnx-pocket-tts-int8-2026-01-26",
            hf="k2-fsa/web-assembly-en-tts-pocket",
            ms="csukuangfj/web-assembly-en-tts-pocket",
            cmd="""
            pushd $model_name

            mv -v lm_flow.int8.onnx ../
            mv -v lm_main.int8.onnx ../
            mv -v encoder.onnx ../
            mv -v decoder.int8.onnx ../
            mv -v text_conditioner.onnx ../
            mv -v vocab.json ../
            mv -v token_scores.json ../
            popd

            git checkout .
            sed -i.bak 's/let modelType = 0/let modelType = 5/g' ../sherpa-onnx-tts.js
            rm -rf $model_name
            git diff
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
        "./run-tts.sh",
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
