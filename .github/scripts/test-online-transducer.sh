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
ls -lh *.onnx
popd

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/8k.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  $wave
done

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run LSTM transducer (Chinese)"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-lstm-zh-2023-02-20
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
ls -lh *.onnx
popd

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/8k.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-11-avg-1.onnx \
  --decoder=$repo/decoder-epoch-11-avg-1.onnx \
  --joiner=$repo/joiner-epoch-11-avg-1.onnx \
  --num-threads=2 \
  $wave
done

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-11-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-11-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-11-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run streaming Zipformer transducer (English)"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
ls -lh *.onnx
popd

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/8k.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  $wave
done

# test int8
#
for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run streaming Zipformer transducer (Bilingual, Chinese + English)"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
ls -lh *.onnx
popd

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
$repo/test_wavs/3.wav
$repo/test_wavs/8k.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  $wave
done

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

# Decode a URL
if [ $EXE == "sherpa-onnx-ffmpeg" ]; then
  time $EXE \
  $repo/tokens.txt \
  $repo/encoder-epoch-99-avg-1.onnx \
  $repo/decoder-epoch-99-avg-1.onnx \
  $repo/joiner-epoch-99-avg-1.onnx \
  https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/test_wavs/4.wav \
  2
fi

if [ $EXE == "sherpa-onnx-ffmpeg" ]; then
  time $EXE \
  $repo/tokens.txt \
  $repo/encoder-epoch-99-avg-1.int8.onnx \
  $repo/decoder-epoch-99-avg-1.int8.onnx \
  $repo/joiner-epoch-99-avg-1.int8.onnx \
  https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/test_wavs/4.wav \
  2
fi

rm -rf $repo

log "------------------------------------------------------------"
log "Run streaming Conformer transducer (English)"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-conformer-en-2023-05-09
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.onnx"
ls -lh *.onnx
popd

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
)

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  $wave
done

for wave in ${waves[@]}; do
  time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  --num-threads=2 \
  $wave
done

rm -rf $repo
