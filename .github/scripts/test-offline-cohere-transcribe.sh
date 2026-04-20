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

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
rm sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
repo=sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01

log "Start testing ${repo_url}"

log "test en"
time $EXE \
  --tokens=$repo/tokens.txt \
  --cohere-transcribe-encoder=$repo/encoder.int8.onnx \
  --cohere-transcribe-decoder=$repo/decoder.int8.onnx \
  --cohere-transcribe-language=en \
  --num-threads=2 \
  $repo/test_wavs/en.wav

log "test de"
time $EXE \
  --tokens=$repo/tokens.txt \
  --cohere-transcribe-encoder=$repo/encoder.int8.onnx \
  --cohere-transcribe-decoder=$repo/decoder.int8.onnx \
  --cohere-transcribe-language=de \
  --num-threads=2 \
  $repo/test_wavs/de.wav

log "test zh"
time $EXE \
  --tokens=$repo/tokens.txt \
  --cohere-transcribe-encoder=$repo/encoder.int8.onnx \
  --cohere-transcribe-decoder=$repo/decoder.int8.onnx \
  --cohere-transcribe-language=zh \
  --num-threads=2 \
  $repo/test_wavs/zh.wav

rm -rf $repo
