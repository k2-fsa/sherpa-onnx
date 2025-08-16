#!/usr/bin/env bash
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

curl -SL -O https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/parakeet-tdt-0.6b-v2.nemo

curl -SL -O https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav


pip install \
  nemo_toolkit['asr'] \
  "numpy<2" \
  ipython \
  kaldi-native-fbank \
  librosa \
  onnx==1.17.0 \
  onnxmltools==1.13.0 \
  onnxruntime==1.17.1 \
  soundfile

python3 ./export_onnx.py
ls -lh *.onnx

echo "---fp32----"
python3 ./test_onnx.py \
  --encoder ./encoder.int8.onnx \
  --decoder ./decoder.onnx \
  --joiner ./joiner.onnx \
  --tokens ./tokens.txt \
  --wav 2086-149220-0033.wav

echo "---int8----"
python3 ./test_onnx.py \
  --encoder ./encoder.int8.onnx \
  --decoder ./decoder.int8.onnx \
  --joiner ./joiner.int8.onnx \
  --tokens ./tokens.txt \
  --wav 2086-149220-0033.wav

echo "---fp16----"
python3 ./test_onnx.py \
  --encoder ./encoder.fp16.onnx \
  --decoder ./decoder.fp16.onnx \
  --joiner ./joiner.fp16.onnx \
  --tokens ./tokens.txt \
  --wav 2086-149220-0033.wav
