#!/usr/bin/env bash
set -ex

# See also
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html#sherpa-onnx-nemo-parakeet-tdt-0-6b-v2-int8-english
if [ ! -f "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx" ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
    tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
    rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
    ls -lh sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
fi

# Run Rust Nemo Parakeet example
cargo run --example nemo_parakeet -- \
    --wav "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/test_wavs/0.wav" \
    --encoder "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx" \
    --decoder "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx" \
    --joiner "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx" \
    --tokens "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt" \
    --provider cpu \
    --num-threads 2 \
    --debug
