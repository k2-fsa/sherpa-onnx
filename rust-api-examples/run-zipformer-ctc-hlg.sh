#!/usr/bin/env bash
set -ex

# see also
# https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
if [ ! -f "./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2

  tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  ls -lh sherpa-onnx-zipformer-ctc-en-2023-10-02
fi

# Offline CTC decoding with an HLG graph via ctc_fst_decoder_config.
cargo run --example zipformer_ctc_hlg -- \
    --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx \
    --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
    --graph ./sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst \
    --wav ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav
