#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Generate build matrix for test-flutter-vad-asr.yaml.
# Each job builds one demo with one ASR model.

import json

# (model_index, name, tarball filename)
# Language suffix added to name when < 6 chars.
ASR_MODELS = [
    (0, "zipformer-ctc-zh", "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2"),
    (1, "sense-voice", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"),
    (2, "whisper-tiny-en", "sherpa-onnx-whisper-tiny.en.tar.bz2"),
    (3, "nemo-parakeet-v2-en", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"),
    (4, "nemo-parakeet-v3-eu", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"),
    (5, "moonshine-en", "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2"),
    (6, "qwen3-asr-multi", "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2"),
    (7, "funasr-nano", "sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2"),
    (8, "firered-ctc-zh-en", "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2"),
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


def make_entry(demo, idx, name, tarball):
    return {
        "demo": demo,
        "asr_model_index": idx,
        "asr_model_name": name,
        "asr_tarball": tarball,
    }


def main():
    desktop_entries = []
    for demo in DEMOS:
        for idx, name, tarball in ASR_MODELS:
            desktop_entries.append(make_entry(demo, idx, name, tarball))

    android_entries = []
    for demo in DEMOS:
        for idx, name, tarball in ASR_MODELS:
            for abi_cfg in ANDROID_ABIS:
                entry = make_entry(demo, idx, name, tarball)
                entry.update(abi_cfg)
                android_entries.append(entry)

    print(json.dumps({
        "desktop": {"include": desktop_entries},
        "android": {"include": android_entries},
    }))


if __name__ == "__main__":
    main()
