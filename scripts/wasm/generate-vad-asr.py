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
    hf: str  # huggingface space name
    ms: str  # modelscope space name
    short_name: str
    cmd: str = ""


def get_models():
    models = [
        Model(
            model_name="sherpa-onnx-whisper-tiny.en",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-whisper-tiny",
            ms="csukuangfj/web-assembly-vad-asr-sherpa-onnx-en-whisper-tiny",
            short_name="vad-asr-en-whisper_tiny",
            cmd="""
            pushd $model_name
            mv -v tiny.en-encoder.int8.onnx ../whisper-encoder.onnx
            mv -v tiny.en-decoder.int8.onnx ../whisper-decoder.onnx
            mv -v tiny.en-tokens.txt ../tokens.txt
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/Whisper tiny.en supporting English 英文/g' ../index.html
            git diff
            """,
        ),
        Model(
            model_name="sherpa-onnx-moonshine-tiny-en-int8",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-moonshine-tiny",
            ms="csukuangfj/web-assembly-vad-asr-sherpa-onnx-en-moonshine-tiny",
            short_name="vad-asr-en-moonshine_tiny",
            cmd="""
            pushd $model_name
            mv -v preprocess.onnx ../moonshine-preprocessor.onnx
            mv -v encode.int8.onnx ../moonshine-encoder.onnx
            mv -v uncached_decode.int8.onnx ../moonshine-uncached-decoder.onnx
            mv -v cached_decode.int8.onnx ../moonshine-cached-decoder.onnx
            mv -v tokens.txt ../
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/Moonshine tiny supporting English 英文/g' ../index.html
            git diff
            """,
        ),
        Model(
            model_name="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-ja-ko-cantonese-sense-voice",
            ms="csukuangfj/web-assembly-vad-asr-sherpa-onnx-zh-en-jp-ko-cantonese-sense-voice",
            short_name="vad-asr-zh_en_ja_ko_cantonese-sense_voice_small",
            cmd="""
            pushd $model_name
            mv -v model.int8.onnx ../sense-voice.onnx
            mv -v tokens.txt ../
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/SenseVoice Small supporting English, Chinese, Japanese, Korean, Cantonese 中英日韩粤/g' ../index.html
            git diff
            """,
        ),
        Model(
            model_name="sherpa-onnx-paraformer-zh-2023-09-14",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer",
            ms="k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer",
            short_name="vad-asr-zh_en-paraformer_large",
            cmd="""
            pushd $model_name
            mv -v model.int8.onnx ../paraformer.onnx
            mv -v tokens.txt ../
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/Paraformer supporting Chinese, English 中英/g' ../index.html
            git diff
            """,
        ),
        Model(
            model_name="sherpa-onnx-paraformer-zh-small-2024-03-09",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer-small",
            ms="k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer-small",
            short_name="vad-asr-zh_en-paraformer_small",
            cmd="""
            pushd $model_name
            mv -v model.int8.onnx ../paraformer.onnx
            mv -v tokens.txt ../
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/Paraformer-small supporting Chinese, English 中英文/g' ../index.html
            git diff
            """,
        ),
        Model(
            model_name="sherpa-onnx-zipformer-gigaspeech-2023-12-12",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-zipformer-gigaspeech",
            ms="k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-zipformer-gigaspeech",
            short_name="vad-asr-en-zipformer_gigaspeech",
            cmd="""
            pushd $model_name
            mv encoder-epoch-30-avg-1.int8.onnx ../transducer-encoder.onnx
            mv decoder-epoch-30-avg-1.onnx ../transducer-decoder.onnx
            mv joiner-epoch-30-avg-1.int8.onnx ../transducer-joiner.onnx
            mv tokens.txt ../
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/Zipformer supporting English 英语/g' ../index.html
            git diff
            """,
        ),
        Model(
            model_name="icefall-asr-zipformer-wenetspeech-20230615",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-zipformer-wenetspeech",
            ms="k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-zipformer-wenetspeech",
            short_name="vad-asr-zh-zipformer_wenetspeech",
            cmd="""
            pushd $model_name
            mv -v data/lang_char/tokens.txt ../
            mv -v exp/encoder-epoch-12-avg-4.int8.onnx ../transducer-encoder.onnx
            mv -v exp/decoder-epoch-12-avg-4.onnx ../transducer-decoder.onnx
            mv -v exp/joiner-epoch-12-avg-4.int8.onnx ../transducer-joiner.onnx
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/Zipformer supporting Chinese 中文/g' ../index.html
            git diff
            """,
        ),
        Model(
            model_name="sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-ja-zipformer",
            ms="csukuangfj/web-assembly-vad-asr-sherpa-onnx-ja-zipformer",
            short_name="vad-asr-ja-zipformer_reazonspeech",
            cmd="""
            pushd $model_name
            mv encoder-epoch-99-avg-1.int8.onnx ../transducer-encoder.onnx
            mv decoder-epoch-99-avg-1.onnx ../transducer-decoder.onnx
            mv joiner-epoch-99-avg-1.int8.onnx ../transducer-joiner.onnx
            mv tokens.txt ../
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/Zipformer supporting Japanese 日语/g' ../index.html
            git diff
            """,
        ),
        Model(
            model_name="sherpa-onnx-zipformer-thai-2024-06-20",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-th-zipformer",
            ms="csukuangfj/web-assembly-vad-asr-sherpa-onnx-th-zipformer",
            short_name="vad-asr-th-zipformer_gigaspeech2",
            cmd="""
            pushd $model_name
            mv encoder-epoch-12-avg-5.int8.onnx ../transducer-encoder.onnx
            mv decoder-epoch-12-avg-5.onnx ../transducer-decoder.onnx
            mv joiner-epoch-12-avg-5.int8.onnx ../transducer-joiner.onnx
            mv tokens.txt ../
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/Zipformer supporting Thai 泰语/g' ../index.html
            git diff
            """,
        ),
        Model(
            model_name="sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04",
            hf="k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-telespeech",
            ms="k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-telespeech",
            short_name="vad-asr-zh-telespeech",
            cmd="""
            pushd $model_name
            mv model.int8.onnx ../telespeech.onnx
            mv tokens.txt ../
            popd
            rm -rf $model_name
            sed -i.bak 's/Zipformer/TeleSpeech-ASR supporting Chinese 多种中文方言/g' ../index.html
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
        "./run-vad-asr.sh",
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
