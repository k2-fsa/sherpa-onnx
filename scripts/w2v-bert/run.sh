#!/usr/bin/env bash
# Copyright (c)  2026  graehl

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

model_id=${MODEL_ID:-proxectonos/w2v-bert-2.0-gl}

pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==2.6.*" \
  "transformers==4.57.*" \
  onnx==1.23.2 \
  onnxruntime==1.31.1 \
  onnxscript \
  "numpy<2" \
  pyarrow==17.0.0 \
  kaldi-native-fbank \
  soundfile

log "Download test waves"
./download_test_wavs.py --num 4

log "Export $model_id"
./export_onnx.py --model-id "$model_id"
ls -lh

cat transcript.txt

log "fp32"
./test_onnx.py --check-batch --model ./model.onnx --tokens ./tokens.txt \
  0.wav 1.wav 2.wav 3.wav

log "int8"
./test_onnx.py --model ./model.int8.onnx --tokens ./tokens.txt \
  0.wav 1.wav 2.wav 3.wav
