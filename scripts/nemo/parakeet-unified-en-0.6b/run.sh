#!/usr/bin/env bash
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/parakeet-unified-en-0.6b.nemo
curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/bias.md
curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/explainability.md
curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/privacy.md
curl -SL -O https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/safety.md

curl -SL -O https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav




pip install \
  "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git" \
  "numpy<2" \
  ipython \
  kaldi-native-fbank \
  librosa \
  onnx \
  onnxruntime \
  soundfile

python3 ./export_onnx.py
ls -lh *.onnx

echo "---int8----"
python3 ./test_onnx.py \
  --encoder ./encoder.int8.onnx \
  --decoder ./decoder.int8.onnx \
  --joiner ./joiner.int8.onnx \
  --tokens ./tokens.txt \
  --wav 2086-149220-0033.wav

echo "---fp32----"
python3 ./test_onnx.py \
  --encoder ./encoder.int8.onnx \
  --decoder ./decoder.onnx \
  --joiner ./joiner.onnx \
  --tokens ./tokens.txt \
  --wav 2086-149220-0033.wav
