#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run zipformer for audio tagging                             "
log "------------------------------------------------------------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
tar xvf sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
rm sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
repo=sherpa-onnx-zipformer-audio-tagging-2024-04-09
ls -lh $repo

for w in 1.wav 2.wav 3.wav 4.wav; do
  $EXE \
    --zipformer-model=$repo/model.onnx \
    --labels=$repo/class_labels_indices.csv \
    $repo/test_wavs/$w
done
rm -rf $repo
