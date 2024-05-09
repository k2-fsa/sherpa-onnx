#!/usr/bin/env bash
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

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
  ./export-onnx-transducer.py --model $m
  d=sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-${m}ms
  if [ ! -f $d/encoder.onnx ]; then
    mkdir -p $d
    mv -v encoder.onnx $d/
    mv -v decoder.onnx $d/
    mv -v joiner.onnx $d/
    mv -v tokens.txt $d/
    ls -lh $d
  fi
done

# Now test the exported models

for m in ${ms[@]}; do
  d=sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-${m}ms
  python3 ./test-onnx-transducer.py \
    --encoder $d/encoder.onnx \
    --decoder $d/decoder.onnx \
    --joiner $d/joiner.onnx \
    --tokens $d/tokens.txt \
    --wav ./0.wav
done
