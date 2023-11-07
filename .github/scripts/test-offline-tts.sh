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

# test waves are saved in ./tts
mkdir ./tts

log "------------------------------------------------------------"
log "vits-ljs test"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/vits-ljs
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
ls -lh *.onnx
popd

$EXE \
  --vits-model=$repo/vits-ljs.onnx \
  --vits-lexicon=$repo/lexicon.txt \
  --vits-tokens=$repo/tokens.txt \
  --output-filename=./tts/vits-ljs.wav \
  'liliana, the most beautiful and lovely assistant of our team!'

ls -lh ./tts

rm -rfv $repo

log "------------------------------------------------------------"
log "vits-vctk test"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/vits-vctk
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
ls -lh *.onnx
popd

for sid in 0 10 90; do
  $EXE \
    --vits-model=$repo/vits-vctk.onnx \
    --vits-lexicon=$repo/lexicon.txt \
    --vits-tokens=$repo/tokens.txt \
    --sid=$sid \
    --output-filename=./tts/vits-vctk-${sid}.wav \
    'liliana, the most beautiful and lovely assistant of our team!'
done

rm -rfv $repo

ls -lh tts/

log "------------------------------------------------------------"
log "vits-zh-aishell3"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/vits-zh-aishell3
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
ls -lh *.onnx
popd

for sid in 0 10 90; do
  $EXE \
    --vits-model=$repo/vits-aishell3.onnx \
    --vits-lexicon=$repo/lexicon.txt \
    --vits-tokens=$repo/tokens.txt \
    --sid=$sid \
    --output-filename=./tts/vits-aishell3-${sid}.wav \
    '林美丽最美丽'
done

rm -rfv $repo

ls -lh ./tts/
