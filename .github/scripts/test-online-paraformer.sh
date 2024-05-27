#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

export GIT_CLONE_PROTECTION_ACTIVE=false

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run streaming Paraformer"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
repo=sherpa-onnx-streaming-paraformer-bilingual-zh-en

log "Start testing ${repo_url}"

time $EXE \
  --tokens=$repo/tokens.txt \
  --paraformer-encoder=$repo/encoder.onnx \
  --paraformer-decoder=$repo/decoder.onnx \
  --num-threads=2 \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/3.wav \
  $repo/test_wavs/8k.wav

time $EXE \
  --tokens=$repo/tokens.txt \
  --paraformer-encoder=$repo/encoder.int8.onnx \
  --paraformer-decoder=$repo/decoder.int8.onnx \
  --num-threads=2 \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/3.wav \
  $repo/test_wavs/8k.wav

rm -rf $repo
