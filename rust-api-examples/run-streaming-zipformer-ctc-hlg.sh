#!/usr/bin/env bash
set -ex

# see also
# https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
if [ ! -f "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2

  tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  ls -lh sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18
fi

# Streaming CTC decoding with an HLG graph via ctc_fst_decoder_config.
cargo run --example streaming_zipformer_ctc_hlg -- \
    --model ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
    --tokens ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt \
    --graph ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst \
    --wav ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/8k.wav
