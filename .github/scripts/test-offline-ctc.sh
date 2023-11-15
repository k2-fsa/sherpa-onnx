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
log "Run Wenet models"
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
    --wenet-ctc-model=$repo/model.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  log "test int8 models"
  time $EXE \
    --tokens=$repo/tokens.txt \
    --wenet-ctc-model=$repo/model.int8.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  rm -rf $repo
done

log "------------------------------------------------------------"
log "Run tdnn yesno (Hebrew)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-tdnn-yesno
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
  --sample-rate=8000 \
  --feat-dim=23 \
  \
  --tokens=$repo/tokens.txt \
  --tdnn-model=$repo/model-epoch-14-avg-2.onnx \
  $repo/test_wavs/0_0_0_1_0_0_0_1.wav \
  $repo/test_wavs/0_0_1_0_0_0_1_0.wav \
  $repo/test_wavs/0_0_1_0_0_1_1_1.wav \
  $repo/test_wavs/0_0_1_0_1_0_0_1.wav \
  $repo/test_wavs/0_0_1_1_0_0_0_1.wav \
  $repo/test_wavs/0_0_1_1_0_1_1_0.wav

log "test int8 models"
time $EXE \
  --sample-rate=8000 \
  --feat-dim=23 \
  \
  --tokens=$repo/tokens.txt \
  --tdnn-model=$repo/model-epoch-14-avg-2.int8.onnx \
  $repo/test_wavs/0_0_0_1_0_0_0_1.wav \
  $repo/test_wavs/0_0_1_0_0_0_1_0.wav \
  $repo/test_wavs/0_0_1_0_0_1_1_1.wav \
  $repo/test_wavs/0_0_1_0_1_0_0_1.wav \
  $repo/test_wavs/0_0_1_1_0_0_0_1.wav \
  $repo/test_wavs/0_0_1_1_0_1_1_0.wav

rm -rf $repo

log "------------------------------------------------------------"
log "Run Citrinet (stt_en_citrinet_512, English)"
log "------------------------------------------------------------"

repo_url=http://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-citrinet-512
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
  --nemo-ctc-model=$repo/model.onnx \
  --num-threads=2 \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

time $EXE \
  --tokens=$repo/tokens.txt \
  --nemo-ctc-model=$repo/model.int8.onnx \
  --num-threads=2 \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

rm -rf $repo

log "------------------------------------------------------------"
log "Run Librispeech zipformer CTC H/HL/HLG decoding (English)   "
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-ctc-en-2023-10-02
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
git lfs pull --include "*.fst"
ls -lh
popd

graphs=(
$repo/H.fst
$repo/HL.fst
$repo/HLG.fst
)

for graph in ${graphs[@]}; do
  log "test float32 models with $graph"
  time $EXE \
    --model-type=zipformer2_ctc \
    --ctc.graph=$graph \
    --zipformer-ctc-model=$repo/model.onnx \
    --tokens=$repo/tokens.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav

  log "test int8 models with $graph"
  time $EXE \
    --model-type=zipformer2_ctc \
    --ctc.graph=$graph \
    --zipformer-ctc-model=$repo/model.int8.onnx \
    --tokens=$repo/tokens.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav
done

rm -rf $repo
