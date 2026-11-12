#!/usr/bin/env bash

# Export a *streaming* X-ASR encoder ONNX with a static batch dimension.
#
# This runs inside the icefall CPU docker image (same image as export-non-streaming.sh).
# The result has 116 inputs (x + 115 caches) and 116 outputs, with no dynamic
# dimensions, which is what prepare-streaming-encoder.py expects.
#
# Usage (inside the container):
#   WORK=/sherpa-onnx/axmodel-work-streaming bash export-streaming-onnx.sh
set -ex

WORK=${WORK:-/sherpa-onnx/axmodel-work-streaming}
ICEFAIL_DIR=${ICEFAIL_DIR:-/icefall}
# 960 ms chunk -> chunk-size 48 (frames of 20 ms); left context 2.56 s
CHUNK_SIZE=${CHUNK_SIZE:-48}
LEFT_CONTEXT=${LEFT_CONTEXT:-256}

mkdir -p "$WORK/exp"
pushd "$WORK/exp"
curl -SL -O https://huggingface.co/GilgameshWind/X-ASR-zh-en/resolve/main/streaming_exp/pretrained.pt
ln -sf pretrained.pt epoch-99.pt
curl -SL -O https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000/tokens.txt
popd

pushd "$ICEFAIL_DIR/egs/librispeech/ASR"
python3 ./zipformer/export-onnx-streaming.py \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir "$WORK/exp" \
  --tokens "$WORK/exp/tokens.txt" \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --causal 1 \
  --chunk-size "$CHUNK_SIZE" \
  --left-context-frames "$LEFT_CONTEXT" \
  --dynamic-batch 0 \
  --use-int32-inputs 0 \
  --enable-int8-quantization 0
popd

# A few wavs for the calibration set (the X-ASR test clips + extra Chinese/English)
mkdir -p "$WORK/wavs"
pushd "$WORK/wavs"
for i in 0 1 2 3 4; do
  curl -SL -O "https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/$i.wav"
done
BASE=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/test_wavs
for i in 0 1 2 3; do
  curl -SL -o "extra-$i.wav" "$BASE/$i.wav"
done
popd

rm -f "$WORK/exp/pretrained.pt"
ls -lh "$WORK/exp"/*.onnx
