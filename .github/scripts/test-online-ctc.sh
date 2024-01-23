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
log "Run streaming Zipformer2 CTC                                "
log "------------------------------------------------------------"

url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
repo=$(basename -s .tar.bz2 $url)
curl -SL -O $url
tar xvf $repo.tar.bz2
rm $repo.tar.bz2

log "test fp32"

time $EXE \
  --debug=1 \
  --zipformer2-ctc-model=$repo/ctc-epoch-20-avg-1-chunk-16-left-128.onnx \
  --tokens=$repo/tokens.txt \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav

log "test int8"

time $EXE \
  --debug=1 \
  --zipformer2-ctc-model=$repo/ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
  --tokens=$repo/tokens.txt \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav


log "------------------------------------------------------------"
log "Run streaming Conformer CTC from WeNet"
log "------------------------------------------------------------"
wenet_models=(
sherpa-onnx-zh-wenet-aishell
sherpa-onnx-zh-wenet-aishell2
sherpa-onnx-zh-wenet-wenetspeech
sherpa-onnx-zh-wenet-multi-cn
sherpa-onnx-en-wenet-librispeech
sherpa-onnx-en-wenet-gigaspeech
)
for name in ${wenet_models[@]}; do
  repo_url=https://huggingface.co/csukuangfj/$name
  log "Start testing ${repo_url}"
  repo=$(basename $repo_url)
  log "Download pretrained model and test-data from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  pushd $repo
  git lfs pull --include "*.onnx"
  ls -lh *.onnx
  popd

  log "test float32 models"
  time $EXE \
    --tokens=$repo/tokens.txt \
    --wenet-ctc-model=$repo/model-streaming.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  log "test int8 models"
  time $EXE \
    --tokens=$repo/tokens.txt \
    --wenet-ctc-model=$repo/model-streaming.int8.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  rm -rf $repo
done
