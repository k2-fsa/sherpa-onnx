#!/usr/bin/env bash
set -ex

# Download Nemo Parakeet TDT 0.6b V2 INT8 model if not present
MODEL_DIR="./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
if [ ! -f "$MODEL_DIR/encoder.int8.onnx" ]; then
    echo "Downloading Nemo Parakeet TDT 0.6b V2 INT8 model ..."
    wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
    tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
    rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
    ls -lh "$MODEL_DIR"
fi

# Run Rust Nemo Parakeet example
cargo run --example nemo_parakeet -- \
    --wav "$MODEL_DIR/test_wavs/en.wav" \
    --encoder "$MODEL_DIR/encoder.int8.onnx" \
    --decoder "$MODEL_DIR/decoder.int8.onnx" \
    --joiner "$MODEL_DIR/joiner.int8.onnx" \
    --tokens "$MODEL_DIR/tokens.txt" \
    --provider cpu \
    --num_threads 2 \
    --debug false
