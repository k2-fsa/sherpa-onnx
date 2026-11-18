#!/usr/bin/env bash
set -ex

repo=english-mfa-v3.1.0
if [ ! -f ./$repo/lexicon.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/$repo.tar.bz2
  tar xvf $repo.tar.bz2
  rm $repo.tar.bz2
fi

cargo run --example forced_alignment_english --   --model-dir ./$repo   --wav ./$repo/en.wav   --text 'ask not what your country can do for you, ask what you can do for your country.'
