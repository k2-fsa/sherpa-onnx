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
log "Run streaming Paraformer"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
ls -lh *.onnx
popd

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
