#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Generate download commands for all ASR models used by the VAD+ASR Flutter demos.
# Used by .github/workflows/test-flutter-vad-asr.yaml

import json

# Each entry: (index, name, tarball filename)
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

# Android ABI configurations.
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


def generate_download_script():
    """Generate shell commands to download and clean up all ASR models."""
    lines = []
    for idx, name, tarball in ASR_MODELS:
        lines.append(f"curl -sSL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{tarball}")
        lines.append(f"tar xjf {tarball}")
        lines.append(f"rm {tarball}")
    # Remove unnecessary files from all model directories.
    lines.append("find . -mindepth 2 -maxdepth 2 -type d -name test_wavs -exec rm -rf {} + 2>/dev/null || true")
    lines.append("find . -mindepth 2 -maxdepth 2 -type f \\( -name '*.md' -o -name '*.txt' ! -name 'tokens.txt' \\) -delete 2>/dev/null || true")
    return "\n".join(lines)


def make_entry(demo, idx, name, tarball):
    return {
        "demo": demo,
        "asr_model_index": idx,
        "asr_model_name": name,
        "asr_tarball": tarball,
    }


def main():
    # Desktop/web matrix: demo × ASR model
    desktop_entries = []
    for demo in [
        "vad-non-streaming-asr-from-file",
        "vad-non-streaming-asr-from-microphone",
    ]:
        for idx, name, tarball in ASR_MODELS:
            desktop_entries.append(make_entry(demo, idx, name, tarball))

    # Android matrix: demo × ASR model × ABI
    android_entries = []
    for demo in [
        "vad-non-streaming-asr-from-file",
        "vad-non-streaming-asr-from-microphone",
    ]:
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
