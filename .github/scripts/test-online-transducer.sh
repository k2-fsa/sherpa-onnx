#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run LSTM transducer (English)"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-lstm-en-2023-02-17

log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
popd

waves=(
$repo/test_wavs/1089-134686-0001.wav
$repo/test_wavs/1221-135766-0001.wav
$repo/test_wavs/1221-135766-0002.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  $repo/tokens.txt \
  $repo/encoder-epoch-99-avg-1.onnx \
  $repo/decoder-epoch-99-avg-1.onnx \
  $repo/joiner-epoch-99-avg-1.onnx \
  $wave \
  2
done

rm -rf $repo
