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
    # We will download
    # https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_name}.tar.bz2
    model_name: str

    # The type of the model, e..g, 0, 1, 2. It is hardcoded in the kotlin code
    idx: int

    # e.g., zh, en, zh_en
    lang: str

    # e.g., whisper, paraformer, zipformer
    short_name: str = ""

    # cmd is used to remove extra file from the model directory
    cmd: str = ""

    rule_fsts: str = ""


# See get_2nd_models() in ./generate-asr-2pass-apk-script.py
def get_models():
    models = [
        Model(
            model_name="sherpa-onnx-whisper-tiny.en",
            idx=2,
            lang="en",
            short_name="whisper_tiny",
            cmd="""
            pushd $model_name
            rm -fv tiny.en-encoder.onnx
            rm -fv tiny.en-decoder.onnx
            rm -rf test_wavs
            rm -fv *.py
            rm -fv requirements.txt
            rm -fv .gitignore
            rm -fv README.md

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-paraformer-zh-2023-09-14",
            idx=0,
            lang="zh",
            short_name="paraformer",
            rule_fsts="itn_zh_number.fst",
            cmd="""
            if [ ! -f itn_zh_number.fst ]; then
              curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
            fi
            pushd $model_name

            rm -fv README.md
            rm -rfv test_wavs
            rm -fv model.onnx

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            idx=15,
            lang="zh_en_ko_ja_yue",
            short_name="sense_voice",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs
            rm -fv model.onnx
            rm -fv *.py

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-paraformer-zh-small-2024-03-09",
            idx=14,
            lang="zh",
            short_name="small_paraformer",
            rule_fsts="itn_zh_number.fst",
            cmd="""
            if [ ! -f itn_zh_number.fst ]; then
              curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
            fi
            pushd $model_name

            rm -fv README.md
            rm -fv *.py
            rm -fv *.yaml
            rm -fv *.mvn
            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="icefall-asr-zipformer-wenetspeech-20230615",
            idx=4,
            lang="zh",
            short_name="zipformer",
            rule_fsts="itn_zh_number.fst",
            cmd="""
            if [ ! -f itn_zh_number.fst ]; then
              curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
            fi
            pushd $model_name

            rm -rfv test_wavs
            rm -fv README.md
            mv -v data/lang_char/tokens.txt ./
            rm -rfv data/lang_char

            mv -v exp/encoder-epoch-12-avg-4.int8.onnx ./
            mv -v exp/decoder-epoch-12-avg-4.onnx ./
            mv -v exp/joiner-epoch-12-avg-4.int8.onnx ./
            rm -rfv exp

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k",
            idx=7,
            lang="be_de_en_es_fr_hr_it_pl_ru_uk",
            short_name="fast_conformer_ctc_20k",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-nemo-fast-conformer-ctc-en-24500",
            idx=8,
            lang="en",
            short_name="fast_conformer_ctc_24500",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-nemo-fast-conformer-ctc-en-de-es-fr-14288",
            idx=9,
            lang="en_des_es_fr",
            short_name="fast_conformer_ctc_14288",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-nemo-fast-conformer-ctc-es-1424",
            idx=10,
            lang="es",
            short_name="fast_conformer_ctc_1424",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04",
            idx=11,
            lang="zh",
            short_name="telespeech",
            rule_fsts="itn_zh_number.fst",
            cmd="""
            if [ ! -f itn_zh_number.fst ]; then
              curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
            fi
            pushd $model_name

            rm -rfv test_wavs
            rm -fv test.py

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-zipformer-thai-2024-06-20",
            idx=12,
            lang="th",
            short_name="zipformer",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs
            rm -fv README.md
            rm -fv bpe.model

            rm -fv encoder-epoch-12-avg-5.onnx
            rm -fv decoder-epoch-12-avg-5.int8.onnx
            rm joiner-epoch-12-avg-5.onnx

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-zipformer-korean-2024-06-24",
            idx=13,
            lang="ko",
            short_name="zipformer",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs
            rm -fv README.md
            rm -fv bpe.model

            rm -fv encoder-epoch-99-avg-1.onnx
            rm -fv decoder-epoch-99-avg-1.int8.onnx
            rm -fv joiner-epoch-99-avg-1.onnx

            ls -lh

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
        "./build-apk-vad-asr.sh",
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
