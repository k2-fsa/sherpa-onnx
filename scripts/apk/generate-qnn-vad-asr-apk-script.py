#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path

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

    use_hr: bool = False


# See get_2nd_models() in ./generate-asr-2pass-apk-script.py
def get_models():
    models = [
        Model(
            model_name="sherpa-onnx-qnn-5-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9000,
            lang="zh_en_ko_ja_yue",
            short_name="5-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-8-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9001,
            lang="zh_en_ko_ja_yue",
            short_name="8-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9002,
            lang="zh_en_ko_ja_yue",
            short_name="10-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-13-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9003,
            lang="zh_en_ko_ja_yue",
            short_name="13-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-15-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9004,
            lang="zh_en_ko_ja_yue",
            short_name="15-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-18-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9005,
            lang="zh_en_ko_ja_yue",
            short_name="18-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-20-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9006,
            lang="zh_en_ko_ja_yue",
            short_name="20-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-23-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9007,
            lang="zh_en_ko_ja_yue",
            short_name="23-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-25-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9008,
            lang="zh_en_ko_ja_yue",
            short_name="25-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-28-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9009,
            lang="zh_en_ko_ja_yue",
            short_name="28-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-30-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9010,
            lang="zh_en_ko_ja_yue",
            short_name="30-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-5-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9011,
            lang="zh",
            short_name="5-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-8-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9012,
            lang="zh",
            short_name="8-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-10-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9013,
            lang="zh",
            short_name="10-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-13-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9014,
            lang="zh",
            short_name="13-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-15-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9015,
            lang="zh",
            short_name="15-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-18-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9016,
            lang="zh",
            short_name="16-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-20-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9017,
            lang="zh",
            short_name="20-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-23-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9018,
            lang="zh",
            short_name="23-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-25-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9019,
            lang="zh",
            short_name="25-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-28-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9020,
            lang="zh",
            short_name="28-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-30-seconds-zipformer-ctc-zh-2025-07-03-int8",
            idx=9021,
            lang="zh",
            short_name="30-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

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
        "./build-apk-qnn-vad-asr-simulate-streaming.sh",
    ]
    for filename in filename_list:
        environment = jinja2.Environment()
        if not Path(f"{filename}.in").is_file():
            print(f"skip {filename}")
            continue

        with open(f"{filename}.in") as f:
            s = f.read()
        template = environment.from_string(s)

        s = template.render(**d)
        with open(filename, "w") as f:
            print(s, file=f)


if __name__ == "__main__":
    main()
