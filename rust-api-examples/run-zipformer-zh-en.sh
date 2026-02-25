#!/usr/bin/env bash
set -ex

# see also
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#sherpa-onnx-zipformer-zh-en-2023-11-22-chinese-english
if [ ! -f "./sherpa-onnx-zipformer-zh-en-2023-11-22/encoder-epoch-34-avg-19.int8.onnx" ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2
    tar xvf sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2
    rm sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2
    ls -lh sherpa-onnx-zipformer-zh-en-2023-11-22
fi

# Run Zipformer transducer
cargo run --example zipformer -- \
    --wav "./sherpa-onnx-zipformer-zh-en-2023-11-22/test_wavs/0.wav" \
    --encoder "./sherpa-onnx-zipformer-zh-en-2023-11-22/encoder-epoch-34-avg-19.int8.onnx" \
    --decoder "./sherpa-onnx-zipformer-zh-en-2023-11-22/decoder-epoch-34-avg-19.onnx" \
    --joiner "./sherpa-onnx-zipformer-zh-en-2023-11-22/joiner-epoch-34-avg-19.int8.onnx" \
    --tokens "./sherpa-onnx-zipformer-zh-en-2023-11-22/tokens.txt" \
    --provider cpu \
    --num-threads 2 \
    --debug
