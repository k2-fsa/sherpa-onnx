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


def get_models():
    models = [
        Model(
            model_name="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
            idx=8,
            lang="bilingual_zh_en",
            short_name="zipformer",
            rule_fsts="itn_zh_number.fst",
            cmd="""
            if [ ! -f itn_zh_number.fst ]; then
              curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
            fi
            pushd $model_name
            rm -fv decoder-epoch-99-avg-1.int8.onnx
            rm -fv encoder-epoch-99-avg-1.onnx
            rm -fv joiner-epoch-99-avg-1.onnx

            rm -fv *.sh
            rm -fv bpe.model
            rm -fv README.md
            rm -fv .gitattributes
            rm -fv *state*
            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-en-2023-06-26",
            idx=6,
            lang="en",
            short_name="zipformer2",
            cmd="""
            pushd $model_name
            rm -fv encoder-epoch-99-avg-1-chunk-16-left-128.onnx
            rm -fv decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
            rm -fv joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx

            rm -fv README.md
            rm -fv bpe.model
            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="icefall-asr-zipformer-streaming-wenetspeech-20230615",
            idx=3,
            lang="zh",
            short_name="zipformer2",
            rule_fsts="itn_zh_number.fst",
            cmd="""
            if [ ! -f itn_zh_number.fst ]; then
              curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
            fi
            pushd $model_name
            rm -fv exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx
            rm -fv exp/decoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx
            rm -fv exp/joiner-epoch-12-avg-4-chunk-16-left-128.int8.onnx

            rm -fv data/lang_char/lexicon.txt
            rm -fv data/lang_char/words.txt
            rm -rfv test_wavs
            rm -fv README.md

            ls -lh exp/
            ls -lh data/lang_char

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-fr-2023-04-14",
            idx=7,
            lang="fr",
            short_name="zipformer",
            cmd="""
            pushd $model_name
            rm -fv encoder-epoch-29-avg-9-with-averaged-model.onnx
            rm -fv decoder-epoch-29-avg-9-with-averaged-model.int8.onnx
            rm -fv joiner-epoch-29-avg-9-with-averaged-model.int8.onnx

            rm -fv *.sh
            rm -rfv test_wavs
            rm README.md

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
            idx=9,
            lang="zh",
            short_name="small_zipformer",
            rule_fsts="itn_zh_number.fst",
            cmd="""
            if [ ! -f itn_zh_number.fst ]; then
              curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
            fi
            pushd $model_name
            rm -fv encoder-epoch-99-avg-1.onnx
            rm -fv decoder-epoch-99-avg-1.int8.onnx
            rm -fv joiner-epoch-99-avg-1.onnx

            rm -fv *.sh
            rm -rf test_wavs
            rm README.md

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
            idx=10,
            lang="en",
            short_name="small_zipformer",
            cmd="""
            pushd $model_name
            rm -fv encoder-epoch-99-avg-1.onnx
            rm -fv decoder-epoch-99-avg-1.int8.onnx
            rm -fv joiner-epoch-99-avg-1.onnx

            rm -fv *.sh
            rm -rf test_wavs
            rm README.md

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms",
            idx=11,
            lang="en",
            short_name="nemo_ctc_80ms",
            cmd="""
            pushd $model_name
            rm -rf test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms",
            idx=12,
            lang="en",
            short_name="nemo_ctc_480ms",
            cmd="""
            pushd $model_name
            rm -rf test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-1040ms",
            idx=13,
            lang="en",
            short_name="nemo_ctc_1040ms",
            cmd="""
            pushd $model_name
            rm -rf test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-korean-2024-06-16",
            idx=14,
            lang="ko",
            short_name="zipformer",
            cmd="""
            pushd $model_name
            rm -fv decoder-epoch-99-avg-1.int8.onnx
            rm -fv encoder-epoch-99-avg-1.onnx
            rm -fv joiner-epoch-99-avg-1.onnx

            rm -fv bpe.model
            rm -fv README.md
            rm -fv .gitattributes
            rm -rfv test_wavs

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
        "./build-apk-asr.sh",
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
