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

names=(
tiny
base
)

for name in ${names[@]}; do
  log "------------------------------------------------------------"
  log "Run $name"
  log "------------------------------------------------------------"

  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-$name.tar.bz2
  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-$name-en-int8.tar.bz2
  curl -SL -O $repo_url
  tar xvf sherpa-onnx-moonshine-$name-en-int8.tar.bz2
  rm sherpa-onnx-moonshine-$name-en-int8.tar.bz2
  repo=sherpa-onnx-moonshine-$name-en-int8
  log "Start testing ${repo_url}"

  log "test int8 onnx"

  time $EXE \
    --moonshine-preprocessor=$repo/preprocess.onnx \
    --moonshine-encoder=$repo/encode.int8.onnx \
    --moonshine-uncached-decoder=$repo/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=$repo/cached_decode.int8.onnx \
    --tokens=$repo/tokens.txt \
    --num-threads=2 \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  rm -rf $repo
done
