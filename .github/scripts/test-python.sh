#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-lstm-en-2023-02-17

log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
popd

python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"
sherpa_onnx_version=$(python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)")

echo "sherpa_onnx version: $sherpa_onnx_version"

pwd
ls -lh

ls -lh $repo

python3 python-api-examples/decode-file.py
