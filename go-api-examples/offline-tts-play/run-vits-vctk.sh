#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -d vits-vctk ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2
  tar xvf vits-vctk.tar.bz2
  rm vits-vctk.tar.bz2
fi

go mod tidy
go build

for sid in 0 10 108; do
./offline-tts-play \
  --vits-model=./vits-vctk/vits-vctk.onnx \
  --vits-lexicon=./vits-vctk/lexicon.txt \
  --vits-tokens=./vits-vctk/tokens.txt \
  --sid=0 \
  --debug=1 \
  'Ask not what your country can do for you; ask what you can do for your country.'
done
