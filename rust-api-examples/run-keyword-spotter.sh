#!/usr/bin/env bash
set -ex

repo=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile
if [ ! -f ./$repo/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$repo.tar.bz2
  tar xvf $repo.tar.bz2
  rm $repo.tar.bz2
fi

cargo run --example keyword_spotter --   --wav ./$repo/test_wavs/3.wav   --encoder ./$repo/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx   --decoder ./$repo/decoder-epoch-12-avg-2-chunk-16-left-64.onnx   --joiner ./$repo/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx   --tokens ./$repo/tokens.txt   --keywords-file ./$repo/test_wavs/test_keywords.txt   --provider cpu   --num-threads 1
