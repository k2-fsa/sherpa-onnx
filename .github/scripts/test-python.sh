#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p /tmp/icefall-models
dir=/tmp/icefall-models

log "Test streaming transducer models"

pushd $dir
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

log "Start testing ${repo_url}"
repo=$dir/$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
cd $repo
git lfs pull --include "*.onnx"
popd

python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"
sherpa_onnx_version=$(python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)")

echo "sherpa_onnx version: $sherpa_onnx_version"

pwd
ls -lh

ls -lh $repo

python3 ./python-api-examples/online-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/3.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/online-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/3.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_online_recognizer.py --verbose

log "Test non-streaming transducer models"

pushd $dir
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-en-2023-04-01

log "Start testing ${repo_url}"
repo=$dir/$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
cd $repo
git lfs pull --include "*.onnx"
popd

ls -lh $repo

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

rm -rf $repo

log "Test non-streaming paraformer models"

pushd $dir
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28

log "Start testing ${repo_url}"
repo=$dir/$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
cd $repo
git lfs pull --include "*.onnx"
popd

ls -lh $repo

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --paraformer=$repo/model.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --paraformer=$repo/model.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

rm -rf $repo

log "Test non-streaming NeMo CTC models"

pushd $dir
repo_url=http://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-citrinet-512

log "Start testing ${repo_url}"
repo=$dir/$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
cd $repo
git lfs pull --include "*.onnx"
popd

ls -lh $repo

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --nemo-ctc=$repo/model.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --nemo-ctc=$repo/model.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

rm -rf $repo
