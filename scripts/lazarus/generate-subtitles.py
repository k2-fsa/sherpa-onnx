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
            model_name="sherpa-onnx-moonshine-tiny-en-int8",
            lang="en",
            short_name="moonshine_tiny",
            cmd="""
            pushd $model_name
            mv -v preprocess.onnx moonshine-preprocessor.onnx
            mv -v encode.int8.onnx moonshine-encoder.onnx
            mv -v uncached_decode.int8.onnx moonshine-uncached-decoder.onnx
            mv -v cached_decode.int8.onnx moonshine-cached-decoder.onnx

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
        Model(
            model_name="sherpa-onnx-paraformer-zh-2023-09-14",
            lang="zh_en",
            short_name="paraformer_2023_09_14",
            cmd="""
            pushd $model_name
            rm -fv model.onnx
            mv -v model.int8.onnx paraformer.onnx
            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-paraformer-zh-small-2024-03-09",
            lang="zh_en",
            short_name="paraformer_small_2024_03_09",
            cmd="""
            pushd $model_name
            rm -fv model.onnx
            mv -v model.int8.onnx paraformer.onnx
            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-zipformer-gigaspeech-2023-12-12",
            lang="en",
            short_name="zipformer_gigaspeech_2023_12_12",
            cmd="""
            pushd $model_name
            mv encoder-epoch-30-avg-1.int8.onnx transducer-encoder.onnx
            mv decoder-epoch-30-avg-1.onnx transducer-decoder.onnx
            mv joiner-epoch-30-avg-1.int8.onnx transducer-joiner.onnx

            rm -fv encoder-epoch-30-avg-1.onnx
            rm -fv decoder-epoch-30-avg-1.int8.onnx
            rm -fv joiner-epoch-30-avg-1.onnx

            popd
            """,
        ),
        Model(
            model_name="icefall-asr-zipformer-wenetspeech-20230615",
            lang="zh",
            short_name="zipformer_wenetspeech",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs
            rm -fv README.md
            mv -v data/lang_char/tokens.txt ./
            rm -rfv data/lang_char

            mv -v exp/encoder-epoch-12-avg-4.int8.onnx ./
            mv -v exp/decoder-epoch-12-avg-4.onnx ./
            mv -v exp/joiner-epoch-12-avg-4.int8.onnx ./
            rm -rfv exp

            mv -v encoder-epoch-12-avg-4.int8.onnx transducer-encoder.onnx
            mv -v decoder-epoch-12-avg-4.onnx transducer-decoder.onnx
            mv -v joiner-epoch-12-avg-4.int8.onnx transducer-joiner.onnx

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
            lang="ja",
            short_name="zipformer_reazonspeech_2024_08_01",
            cmd="""
            pushd $model_name
            mv encoder-epoch-99-avg-1.int8.onnx transducer-encoder.onnx
            mv decoder-epoch-99-avg-1.onnx transducer-decoder.onnx
            mv joiner-epoch-99-avg-1.int8.onnx transducer-joiner.onnx

            rm -fv encoder-epoch-99-avg-1.onnx
            rm -fv decoder-epoch-99-avg-1.int8.onnx
            rm -fv joiner-epoch-99-avg-1.onnx

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-zipformer-thai-2024-06-20",
            lang="th",
            short_name="zipformer_gigaspeech2",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs
            rm -fv README.md
            rm -fv bpe.model

            mv encoder-epoch-12-avg-5.int8.onnx transducer-encoder.onnx
            mv decoder-epoch-12-avg-5.onnx transducer-decoder.onnx
            mv joiner-epoch-12-avg-5.int8.onnx transducer-joiner.onnx

            rm -fv encoder-epoch-12-avg-5.onnx
            rm -fv decoder-epoch-12-avg-5.int8.onnx
            rm -fv joiner-epoch-12-avg-5.onnx

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04",
            lang="zh",
            short_name="telespeech_ctc",
            cmd="""
            pushd $model_name

            mv model.int8.onnx telespeech.onnx
            rm -fv model.onnx

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
