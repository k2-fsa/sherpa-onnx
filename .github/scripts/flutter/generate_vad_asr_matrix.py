#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Generate build matrix for test-flutter-vad-asr.yaml.
# Each job builds one demo with one ASR model.

import json

# (model_index, name, tarball filename, cleanup_cmd)
# cleanup_cmd runs inside the model directory after extraction to remove
# files the app does not need (test_wavs, non-int8 weights, scripts, etc.).
# The model directory name is the tarball name without .tar.bz2.
ASR_MODELS = [
    (
        0,
        "zipformer-ctc-zh",
        "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2",
        """
        rm -rf test_wavs
        rm -fv bbpe.model
        rm -fv README.md
        ls -lh
        """,
    ),
    (
        1,
        "sense-voice",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
        """
        rm -rf test_wavs
        rm -fv export-onnx.py
        rm -fv LICENSE
        rm -fv README.md
        ls -lh
        """,
    ),
    (
        2,
        "whisper-tiny-en",
        "sherpa-onnx-whisper-tiny.en.tar.bz2",
        """
        rm -fv tiny.en-encoder.onnx
        rm -fv tiny.en-decoder.onnx
        rm -rf test_wavs
        rm -fv *.py
        rm -fv requirements.txt
        rm -fv .gitignore
        rm -fv README.md
        ls -lh
        """,
    ),
    (
        3,
        "nemo-parakeet-v2-en",
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        """
        rm -rf test_wavs
        ls -lh
        """,
    ),
    (
        4,
        "nemo-parakeet-v3-eu",
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        """
        rm -rf test_wavs
        ls -lh
        """,
    ),
    (
        5,
        "moonshine-en",
        "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2",
        """
        rm -rf test_wavs
        rm -fv LICENSE
        ls -lh
        """,
    ),
    (
        6,
        "qwen3-asr-multi",
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2",
        """
        rm -rf test_wavs
        rm -fv README.md
        rm -fv tokenizer/merges.txt
        rm -fv tokenizer/vocab.json
        rm -fv tokenizer/tokenizer_config.json
        ls -lh
        """,
    ),
    (
        7,
        "funasr-nano",
        "sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2",
        """
        rm -rf test_wavs
        rm -fv README.md
        rm -fv Qwen3-0.6B/merges.txt
        rm -fv Qwen3-0.6B/vocab.json
        ls -lh
        """,
    ),
    (
        8,
        "firered-ctc-zh-en",
        "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2",
        """
        rm -rf test_wavs
        rm -fv README.md
        ls -lh
        """,
    ),
]

DEMOS = [
    "vad-non-streaming-asr-from-file",
    "vad-non-streaming-asr-from-microphone",
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


def make_entry(demo, idx, name, tarball, cleanup_cmd):
    return {
        "demo": demo,
        "asr_model_index": idx,
        "asr_model_name": name,
        "asr_tarball": tarball,
        "asr_cleanup_cmd": cleanup_cmd,
    }


def main():
    desktop_entries = []
    for demo in DEMOS:
        for idx, name, tarball, cmd in ASR_MODELS:
            desktop_entries.append(make_entry(demo, idx, name, tarball, cmd))

    android_entries = []
    for demo in DEMOS:
        for idx, name, tarball, cmd in ASR_MODELS:
            for abi_cfg in ANDROID_ABIS:
                entry = make_entry(demo, idx, name, tarball, cmd)
                entry.update(abi_cfg)
                android_entries.append(entry)

    print(json.dumps({
        "desktop": {"include": desktop_entries},
        "android": {"include": android_entries},
    }))


if __name__ == "__main__":
    main()
