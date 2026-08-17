#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Generate download commands for all ASR models used by the VAD+ASR Flutter demos.
# Used by .github/workflows/test-flutter-vad-asr.yaml

import json

# Each entry: (url, directory_name)
# The url is a .tar.bz2 that extracts to directory_name/
ASR_MODELS = [
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2",
        "sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2",
        "sherpa-onnx-whisper-tiny.en",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2",
        "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2",
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2",
        "sherpa-onnx-funasr-nano-int8-2025-12-30",
    ),
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2",
        "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25",
    ),
]


def generate_download_script():
    """Generate shell commands to download all ASR models."""
    lines = []
    for url, dirname in ASR_MODELS:
        filename = url.rsplit("/", 1)[-1]
        lines.append(f"curl -sSL -O {url}")
        lines.append(f"tar xjf {filename}")
        lines.append(f"rm {filename}")
    return "\n".join(lines)


def main():
    print(generate_download_script())


if __name__ == "__main__":
    main()
