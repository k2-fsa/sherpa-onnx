#!/usr/bin/env bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
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
tiny.en
base.en
small.en
medium.en
tiny
base
small
medium
distil-medium.en
distil-small.en
)

for name in ${names[@]}; do
  log "------------------------------------------------------------"
  log "Run $name"
  log "------------------------------------------------------------"

  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-$name.tar.bz2
  download_and_extract $repo_url
  repo=sherpa-onnx-whisper-$name
  log "Start testing ${repo_url}"

  log "test fp32 onnx"

  time $EXE \
    --tokens=$repo/${name}-tokens.txt \
    --whisper-encoder=$repo/${name}-encoder.onnx \
    --whisper-decoder=$repo/${name}-decoder.onnx \
    --whisper-tail-paddings=500 \
    --num-threads=2 \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  log "test int8 onnx"

  time $EXE \
    --tokens=$repo/${name}-tokens.txt \
    --whisper-encoder=$repo/${name}-encoder.int8.onnx \
    --whisper-decoder=$repo/${name}-decoder.int8.onnx \
    --whisper-tail-paddings=500 \
    --num-threads=2 \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  rm -rf $repo
done
