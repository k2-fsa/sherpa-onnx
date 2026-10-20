#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Generate build matrix for test-flutter-asr.yaml.
# Each job builds flutter-examples/streaming_asr with one streaming ASR model
# selected (`final type = N;` in lib/streaming_asr.dart).

import json

# (model_index, model_dir_name, tarball filename, rule_fsts, post_cmd)
#
# model_dir_name is both the extracted directory in assets/ and the name that
# goes into the released zip (flutter-streaming_asr-<model_dir_name>-<platform>.zip).
# post_cmd runs inside assets/ after extraction (downloads the optional rule
# fst and strips files the app does not need).
STREAMING_ASR_MODELS = [
    (
        0,
        "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
        "itn_zh_number.fst",
        """
        if [ ! -f itn_zh_number.fst ]; then
          curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
        fi
        pushd sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
        rm -fv encoder-epoch-99-avg-1.onnx
        rm -fv decoder-epoch-99-avg-1.int8.onnx
        rm -fv joiner-epoch-99-avg-1.int8.onnx
        rm -fv *.sh bpe.model README.md .gitattributes *state*
        rm -rfv test_wavs
        ls -lh
        popd
        """,
    ),
    (
        1,
        "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "",
        """
        pushd sherpa-onnx-streaming-zipformer-en-2023-06-26
        rm -fv encoder-epoch-99-avg-1-chunk-16-left-128.onnx
        rm -fv decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx
        rm -fv joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx
        rm -fv *.sh bpe.model README.md .gitattributes *state*
        rm -rfv test_wavs
        ls -lh
        popd
        """,
    ),
    (
        2,
        "icefall-asr-zipformer-streaming-wenetspeech-20230615",
        "icefall-asr-zipformer-streaming-wenetspeech-20230615.tar.bz2",
        "itn_zh_number.fst",
        """
        if [ ! -f itn_zh_number.fst ]; then
          curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
        fi
        pushd icefall-asr-zipformer-streaming-wenetspeech-20230615
        rm -fv exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx
        rm -fv exp/decoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx
        rm -fv exp/joiner-epoch-12-avg-4-chunk-16-left-128.int8.onnx
        rm -fv *.sh bpe.model README.md .gitattributes *state*
        rm -rfv test_wavs
        ls -lh
        popd
        """,
    ),
    (
        3,
        "sherpa-onnx-streaming-zipformer-fr-2023-04-14",
        "sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2",
        "",
        """
        pushd sherpa-onnx-streaming-zipformer-fr-2023-04-14
        rm -fv encoder-epoch-12-avg-5.onnx
        rm -fv decoder-epoch-12-avg-5.int8.onnx
        rm -fv joiner-epoch-12-avg-5.int8.onnx
        rm -fv *.sh bpe.model README.md .gitattributes *state*
        rm -rfv test_wavs
        ls -lh
        popd
        """,
    ),
    (
        4,
        "sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05",
        "sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2",
        "",
        """
        pushd sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
        rm -fv encoder.onnx joiner.onnx decoder.int8.onnx
        rm -fv bpe.model test_onnx.py README.md .gitattributes *state*
        rm -rfv test_wavs
        ls -lh
        popd
        """,
    ),
    (
        5,
        "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
        "",
        """
        pushd sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
        rm -fv encoder.onnx decoder.onnx joiner.onnx
        rm -fv *.py README.md .gitattributes *state*
        rm -rfv test_wavs
        ls -lh
        popd
        """,
    ),
    (
        6,
        "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05",
        "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2",
        "",
        """
        pushd sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
        rm -fv encoder.onnx joiner.onnx decoder.int8.onnx
        rm -fv bpe.model test_onnx.py README.md .gitattributes *state*
        rm -rfv test_wavs
        ls -lh
        popd
        """,
    ),
    (
        7,
        "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11",
        "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2",
        "",
        """
        pushd sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11
        rm -fv encoder.onnx decoder.onnx joiner.onnx
        rm -fv *.py README.md .gitattributes *state*
        rm -rfv test_wavs
        ls -lh
        popd
        """,
    ),
]

ANDROID_ABIS = [
    {
        "abi": "arm64-v8a",
        "target_platform": "android-arm64",
        "build_script": "build-android-arm64-v8a.sh",
        "build_dir": "build-android-arm64-v8a",
        "plugin": "sherpa_onnx_android_arm64",
        "plugin_key": "sherpa_onnx_android_arm64",
    },
    {
        "abi": "armeabi-v7a",
        "target_platform": "android-arm",
        "build_script": "build-android-armv7-eabi.sh",
        "build_dir": "build-android-armv7-eabi",
        "plugin": "sherpa_onnx_android_armeabi",
        "plugin_key": "sherpa_onnx_android_armeabi",
    },
    {
        "abi": "x86_64",
        "target_platform": "android-x64",
        "build_script": "build-android-x86-64.sh",
        "build_dir": "build-android-x86-64",
        "plugin": "sherpa_onnx_android_x86_64",
        "plugin_key": "sherpa_onnx_android_x86_64",
    },
]


def make_entry(idx, name, tarball, rule_fsts, post_cmd):
    return {
        "demo": "streaming_asr",
        "asr_model_index": idx,
        "asr_model_name": name,
        "asr_tarball": tarball,
        "asr_rule_fsts": rule_fsts,
        "asr_post_cmd": post_cmd,
    }


def main():
    desktop_entries = []
    android_entries = []
    for idx, name, tarball, rule_fsts, post_cmd in STREAMING_ASR_MODELS:
        desktop_entries.append(make_entry(idx, name, tarball, rule_fsts, post_cmd))
        for abi_cfg in ANDROID_ABIS:
            entry = make_entry(idx, name, tarball, rule_fsts, post_cmd)
            entry.update(abi_cfg)
            android_entries.append(entry)

    print(json.dumps({
        "desktop": {"include": desktop_entries},
        "android": {"include": android_entries},
    }))


if __name__ == "__main__":
    main()
