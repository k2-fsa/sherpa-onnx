#!/usr/bin/env bash

# Container-side half of the streaming axmodel pipeline.
#
# Runs inside the icefall CPU image and produces, in $WORK:
#   encoder-streaming-ax650.onnx   prepared graph (116 inputs / 116 outputs)
#   calib/                          one calibration tar per input tensor
#
# Usage: WORK=/sherpa-onnx/axmodel-work-streaming bash prepare-streaming-axmodel.sh
set -ex

WORK=${WORK:-/sherpa-onnx/axmodel-work-streaming}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WORK

bash "$SCRIPT_DIR/export-streaming-onnx.sh"

ENC_SRC=$(ls "$WORK"/exp/encoder-*-chunk-*-left-*.onnx | head -1)
python3 -m pip install -q onnx onnxruntime kaldi-native-fbank

python3 "$SCRIPT_DIR/prepare-streaming-encoder.py" \
  --src "$ENC_SRC" \
  --dst "$WORK/encoder-streaming-ax650.onnx" \
  --freeze-input processed_lens=256 \
  --drop-output new_processed_lens

python3 "$SCRIPT_DIR/make-streaming-calibration-set.py" \
  --encoder "$WORK/encoder-streaming-ax650.onnx" \
  --wav-dir "$WORK/wavs" \
  --output "$WORK/calib" \
  --num-frames 109 \
  --hop 96 \
  --num-samples 16

ls -lh "$WORK"/encoder-streaming-ax650.onnx
