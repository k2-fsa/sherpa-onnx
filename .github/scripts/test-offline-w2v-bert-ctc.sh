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

d=sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$d.tar.bz2
tar xvf $d.tar.bz2
rm $d.tar.bz2

log "Decode each wave alone"
for w in 0.wav 1.wav 2.wav 3.wav; do
$EXE \
  --w2v-bert-ctc=./$d/model.int8.onnx \
  --tokens=./$d/tokens.txt \
  ./$d/test_wavs/$w
done

log "Decode the waves as one batch"
$EXE \
  --w2v-bert-ctc=./$d/model.int8.onnx \
  --tokens=./$d/tokens.txt \
  ./$d/test_wavs/0.wav \
  ./$d/test_wavs/1.wav \
  ./$d/test_wavs/2.wav \
  ./$d/test_wavs/3.wav

cat ./$d/test_wavs/transcript.txt

rm -rf $d
