#!/usr/bin/env bash

# Prepare a Pulsar2-friendly encoder ONNX for the X-ASR model.
#
# This script is meant to run **inside the icefall CPU docker image** (it needs
# torch + k2 + icefall for export). It does:
#
#   1. download the X-ASR checkpoint / tokens / test wavs (same files as
#      ../export-non-streaming.sh)
#   2. export a static-shape, non-streaming, fp32 ONNX encoder (10 s window)
#   3. run prepare-encoder.py (fold If / freeze shapes / Log -> Softplus)
#   4. build the calibration set from the test wavs
#
# The compiler (Pulsar2) runs on the host afterwards, see
# ../../../../.github/workflows/export-x-asr-axera.yaml
#
# Usage (inside the container):
#   WORK=/sherpa-onnx/axmodel-work bash prepare-axmodel.sh
set -ex

WORK=${WORK:-/sherpa-onnx/axmodel-work}
ICEFAIL_DIR=${ICEFAIL_DIR:-/icefall}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$WORK"/{exp,wavs}
pushd "$WORK/wavs"
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/0.wav
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/1.wav
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/2.wav
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/3.wav
# 4.wav is ~17 s, so it also covers the 10 s window without any padding
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/4.wav
popd

# The X-ASR test set is only five short clips. Add a few Chinese/English clips
# from the sherpa-onnx streaming zipformer test set so that the calibration
# features cover more speakers and levels (the MinMax ranges are sensitive to
# this: a set of short clips alone gives a much worse INT8 model).
pushd "$WORK/wavs"
BASE=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/test_wavs
for f in 0 1 2 3 4 5 6 7; do
  curl -SL -o "extra-$f.wav" "$BASE/$f.wav"
done
popd

pushd "$WORK/exp"
# Note: this is the training checkpoint from the model's HF repo (2.5 GB, contains
# `model` and `model_avg`). The 639 MB `checkpoint/pretrained.pt` in the GitHub repo
# holds the released/averaged weights instead, and the two are numerically different
# (all 989 tensors differ). Quantizing the GitHub file with the settings below gives
# cosine ~0.972 against its own fp32 ONNX, while this checkpoint gives 0.999, so the
# axmodel pipeline uses this one to match the published axmodel.
curl -SL -O https://huggingface.co/GilgameshWind/X-ASR-zh-en/resolve/main/streaming_exp/pretrained.pt
ln -sf pretrained.pt epoch-99.pt
curl -SL -O https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000/tokens.txt
popd

pushd "$ICEFAIL_DIR/egs/librispeech/ASR"
python3 ./zipformer/export-streaming-as-non-streaming-onnx.py \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir "$WORK/exp" \
  --tokens "$WORK/exp/tokens.txt" \
  --max-len 1000 \
  --dynamic-axes 0 \
  --keep-x-lens 0 \
  --use-int32-inputs 0 \
  --enable-int8-quantization 0 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --causal 1 \
  --chunk-size "-1" \
  --left-context-frames "-1"
popd

# the 2.5 GB checkpoint is no longer needed after the export
rm -f "$WORK/exp/pretrained.pt"

python3 -m pip install -q onnx onnxruntime kaldi-native-fbank

python3 "$SCRIPT_DIR/prepare-encoder.py" \
  --src "$WORK/exp/encoder-epoch-99-avg-1.onnx" \
  --dst "$WORK/encoder-ax650.onnx"

python3 "$SCRIPT_DIR/make-calibration-set.py" \
  --wav-dir "$WORK/wavs" \
  --output "$WORK/x4d.tar.gz" \
  --num-frames 1000 \
  --num-samples 16 \
  --shifts 4 \
  --pad tile \
  --work-dir "$WORK/calib_x4d"

ls -lh "$WORK"/encoder-ax650.onnx "$WORK"/x4d.tar.gz
