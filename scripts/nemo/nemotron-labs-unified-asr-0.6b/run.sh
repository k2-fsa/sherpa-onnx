#!/usr/bin/env bash
# Copyright (c) 2026 Julian Pscheid

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
left_context=${1:-70}
case "$left_context" in
  70|37) ;;
  *) echo "Usage: $0 [70|37]" >&2; exit 1 ;;
esac

pip install \
  "nemo_toolkit[asr] @ https://github.com/NVIDIA/NeMo/archive/43dd97ddf.tar.gz" \
  onnx onnxruntime ipython
python3 "$script_dir/test_export_onnx.py"

export_args=(--left-context "$left_context")
if [ -n "${NEMO_MODEL:-}" ]; then
  model=$(cd "$(dirname "$NEMO_MODEL")" && pwd)/$(basename "$NEMO_MODEL")
  export_args+=(--model "$model")
fi
sherpa_onnx=${SHERPA_ONNX_BIN:-$script_dir/../../../build/bin/sherpa-onnx}
sherpa_onnx=$(cd "$(dirname "$sherpa_onnx")" && pwd)/$(basename "$sherpa_onnx")

mkdir -p "export-L$left_context"
cd "export-L$left_context"
python3 "$script_dir/export_onnx.py" "${export_args[@]}"

curl -fL --retry 3 -o test.wav \
  https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
for language in en en-US auto; do
  "$sherpa_onnx" \
    --encoder=encoder.int8.onnx --decoder=decoder.onnx \
    --joiner=joiner.onnx --tokens=tokens.txt --language="$language" test.wav
done

package=sherpa-onnx-nemotron-labs-unified-asr-0.6b-int8-streaming-1120ms-L$left_context
mkdir -p "../$package/test_wavs"
cp encoder.int8.onnx decoder.onnx joiner.onnx tokens.txt "../$package/"
cp test.wav "../$package/test_wavs/0.wav"
cp "$script_dir/README.md" "../$package/"
revision=023ce907267a09a308a5cb69239447cb3d0c1597
base="https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b/resolve/$revision"
for file in LICENSE.md bias.md explainability.md privacy.md safety.md; do
  curl -fL --retry 3 "$base/$file" -o "../$package/$file"
done
curl -fL --retry 3 "$base/README.md" -o "../$package/MODEL_CARD.md"
cd ..
tar cjf "$package.tar.bz2" "$package"
