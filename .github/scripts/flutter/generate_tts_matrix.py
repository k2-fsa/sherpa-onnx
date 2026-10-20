#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Generate build matrix for test-flutter-tts.yaml.
# Each job builds flutter-examples/tts with one TTS model selected
# (selectedModelIndex in lib/model_config.dart).

import json

# (model_index, model_dir_name, tarball filename, extra_cmd, cleanup_cmd)
#
# model_dir_name is both the extracted directory in assets/ and the name that
# goes into the released zip (flutter-tts-<model_dir_name>-<platform>.zip).
# extra_cmd/cleanup_cmd run inside the model's assets/ staging directory.
TTS_MODELS = [
    (
        0,
        "vits-piper-en_US-amy-low",
        "vits-piper-en_US-amy-low.tar.bz2",
        "",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        1,
        "vits-piper-zh_CN-xiao_ya-medium",
        "vits-piper-zh_CN-xiao_ya-medium.tar.bz2",
        "",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        2,
        "vits-piper-en_US-libritts_r-medium",
        "vits-piper-en_US-libritts_r-medium.tar.bz2",
        "",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        3,
        "vits-inflect-en-nano-v2",
        "vits-inflect-en-nano-v2.tar.bz2",
        "",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        4,
        "kokoro-int8-en-v0_19",
        "kokoro-int8-en-v0_19.tar.bz2",
        "",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        5,
        "kokoro-multi-lang-v1_0",
        "kokoro-multi-lang-v1_0.tar.bz2",
        "",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        6,
        "matcha-icefall-en_US-ljspeech",
        "matcha-icefall-en_US-ljspeech.tar.bz2",
        "curl -sSL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        7,
        "matcha-icefall-zh-en",
        "matcha-icefall-zh-en.tar.bz2",
        "curl -sSL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        8,
        "kitten-nano-en-v0_1-fp16",
        "kitten-nano-en-v0_1-fp16.tar.bz2",
        "",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        9,
        "sherpa-onnx-pocket-tts-int8-2026-01-26",
        "sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2",
        "",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
        """,
    ),
    (
        10,
        "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
        "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
        "",
        """
        rm -rf test_wavs espeak-ng-data
        rm -fv README.md LICENSE *.py
        ls -lh
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


def make_entry(idx, name, tarball, extra_cmd, cleanup_cmd):
    return {
        "demo": "tts",
        "tts_model_index": idx,
        "tts_model_name": name,
        "tts_tarball": tarball,
        "tts_extra_cmd": extra_cmd,
        "tts_cleanup_cmd": cleanup_cmd,
    }


def main():
    desktop_entries = []
    android_entries = []
    for idx, name, tarball, extra_cmd, cleanup_cmd in TTS_MODELS:
        desktop_entries.append(make_entry(idx, name, tarball, extra_cmd, cleanup_cmd))
        for abi_cfg in ANDROID_ABIS:
            entry = make_entry(idx, name, tarball, extra_cmd, cleanup_cmd)
            entry.update(abi_cfg)
            android_entries.append(entry)

    print(json.dumps({
        "desktop": {"include": desktop_entries},
        "android": {"include": android_entries},
    }))


if __name__ == "__main__":
    main()
