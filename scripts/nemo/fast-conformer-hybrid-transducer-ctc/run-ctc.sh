#!/usr/bin/env bash

set -ex

if [ ! -e ./0.wav ]; then
  # curl -SL -O https://hf-mirror.com/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/0.wav
fi

ms=(
80
480
1040
)

for m in ${ms[@]}; do
  ./export-onnx-ctc.py --model $m
  d=sherpa-onnx-nemo-streaming-fast-conformer-ctc-${m}ms
  if [ ! -f $d/model.onnx ]; then
    mkdir -p $d
    mv -v model.onnx $d/
    mv -v tokens.txt $d/
    ls -lh $d
  fi
done

# Now test the exported models

for m in ${ms[@]}; do
  d=sherpa-onnx-nemo-streaming-fast-conformer-ctc-${m}ms
  python3 ./test-onnx-ctc.py \
    --model $d/model.onnx \
    --tokens $d/tokens.txt \
    --wav ./0.wav
done
