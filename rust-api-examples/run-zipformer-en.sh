#!/usr/bin/env bash
set -ex

# see also
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#icefall-asr-multidataset-pruned-transducer-stateless7-2023-05-04-english
if [ ! -f "./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/data/lang_bpe_500/tokens.txt" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04.tar.bz2

  tar xvf icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04.tar.bz2
  rm icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04.tar.bz2
  ls -lh icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04
fi

# Run Zipformer transducer
cargo run --example zipformer -- \
    --wav "./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/test_wavs/1089-134686-0001.wav" \
    --tokens=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/data/lang_bpe_500/tokens.txt \
    --encoder=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/encoder-epoch-30-avg-4.int8.onnx \
    --decoder=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/decoder-epoch-30-avg-4.onnx \
    --joiner=./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/joiner-epoch-30-avg-4.int8.onnx \
    --provider cpu \
    --num-threads 2 \
    --debug
