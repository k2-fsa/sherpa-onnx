#!/usr/bin/env bash
set -ex

# see
# https://k2-fsa.github.io/sherpa/onnx/FireRedAsr/pretrained.html
if [ ! -f ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx ]; then
  curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2

  tar xvf sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
  rm sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
  ls -lh sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25
fi

cargo run --example fire_red_asr_ctc -- \
    --wav ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/test_wavs/1.wav \
    --model ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx \
    --tokens ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt \
    --num-threads 2 \
    --debug
