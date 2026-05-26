#!/usr/bin/env bash
set -ex

if [ ! -f ./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx ]; then
  curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2

  tar xvf sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
  rm sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
  ls -lh sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01
fi

cargo run --example cohere_transcribe -- \
    --wav ./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav \
    --encoder ./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx \
    --decoder ./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx \
    --tokens ./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt \
    --language en \
    --num-threads 2 \
    --debug
