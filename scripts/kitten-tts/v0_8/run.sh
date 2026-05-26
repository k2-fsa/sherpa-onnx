#!/usr/bin/env bash
# Copyright      2026  Xiaomi Corp.

set -ex

python3 - <<'PY'
import importlib.util
import sys

missing = [
    m for m in ("numpy", "onnx") if importlib.util.find_spec(m) is None
]

if missing:
    print(
        "Missing Python packages: "
        + ", ".join(missing)
        + "\nInstall them with:\n"
        + "  python3 -m pip install numpy onnx",
        file=sys.stderr,
    )
    sys.exit(1)
PY

model_name=${1:-KittenML/kitten-tts-nano-0.8-fp32}

if [[ ${model_name} != */* ]]; then
  model_name=KittenML/${model_name}
fi

case ${model_name} in
  *kitten-tts-mini-0.8*)
    onnx_name=kitten_tts_mini_v0_8.onnx
    output_name=model.onnx
    package_dir=kitten-mini-en-v0_8
    ;;
  *kitten-tts-micro-0.8*)
    onnx_name=kitten_tts_micro_v0_8.onnx
    output_name=model.onnx
    package_dir=kitten-micro-en-v0_8
    ;;
  *kitten-tts-nano-0.8-int8*)
    onnx_name=kitten_tts_nano_v0_8.onnx
    output_name=model.int8.onnx
    package_dir=kitten-nano-en-v0_8-int8
    ;;
  *kitten-tts-nano-0.8*)
    onnx_name=kitten_tts_nano_v0_8.onnx
    output_name=model.fp32.onnx
    package_dir=kitten-nano-en-v0_8-fp32
    ;;
  *)
    echo "Unsupported KittenTTS v0.8 model: ${model_name}"
    exit 1
    ;;
esac

base_url=https://huggingface.co/${model_name}/resolve/main

if [ ! -f "${onnx_name}" ]; then
  curl -SL -O "${base_url}/${onnx_name}"
fi

if [ ! -f voices.npz ]; then
  curl -SL -O "${base_url}/voices.npz"
fi

cp "${onnx_name}" "${output_name}"
./generate_voices_bin.py
./generate_tokens.py
./add_meta_data.py --model "./${output_name}" --model-name "${model_name}"

mkdir -p "${package_dir}"
cp "${output_name}" "${package_dir}/"
cp voices.bin tokens.txt "${package_dir}/"

ls -lh
ls -lh "${package_dir}"
