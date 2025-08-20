#!/usr/bin/env bash
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/de.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav

pip install \
  nemo_toolkit['asr'] \
  "numpy<2" \
  ipython \
  kaldi-native-fbank \
  librosa \
  onnx==1.17.0 \
  onnxruntime==1.17.1 \
  onnxscript \
  soundfile

python3 ./export_onnx_180m_flash.py
ls -lh *.onnx


log "-----fp32------"

python3 ./test_180m_flash.py \
  --encoder ./encoder.onnx \
  --decoder ./decoder.onnx \
  --source-lang en \
  --target-lang en \
  --tokens ./tokens.txt \
  --wav ./en.wav

python3 ./test_180m_flash.py \
  --encoder ./encoder.onnx \
  --decoder ./decoder.onnx \
  --source-lang en \
  --target-lang de \
  --tokens ./tokens.txt \
  --wav ./en.wav

python3 ./test_180m_flash.py \
  --encoder ./encoder.onnx \
  --decoder ./decoder.onnx \
  --source-lang de \
  --target-lang de \
  --tokens ./tokens.txt \
  --wav ./de.wav

python3 ./test_180m_flash.py \
  --encoder ./encoder.onnx \
  --decoder ./decoder.onnx \
  --source-lang de \
  --target-lang en \
  --tokens ./tokens.txt \
  --wav ./de.wav


log "-----int8------"

python3 ./test_180m_flash.py \
  --encoder ./encoder.int8.onnx \
  --decoder ./decoder.int8.onnx \
  --source-lang en \
  --target-lang en \
  --tokens ./tokens.txt \
  --wav ./en.wav

python3 ./test_180m_flash.py \
  --encoder ./encoder.int8.onnx \
  --decoder ./decoder.int8.onnx \
  --source-lang en \
  --target-lang de \
  --tokens ./tokens.txt \
  --wav ./en.wav

python3 ./test_180m_flash.py \
  --encoder ./encoder.int8.onnx \
  --decoder ./decoder.int8.onnx \
  --source-lang de \
  --target-lang de \
  --tokens ./tokens.txt \
  --wav ./de.wav

python3 ./test_180m_flash.py \
  --encoder ./encoder.int8.onnx \
  --decoder ./decoder.int8.onnx \
  --source-lang de \
  --target-lang en \
  --tokens ./tokens.txt \
  --wav ./de.wav
