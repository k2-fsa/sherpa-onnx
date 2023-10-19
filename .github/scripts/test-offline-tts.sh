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

log "vits-ljs test"

wget -qq https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx
wget -qq https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt
wget -qq https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt

$EXE \
  --vits-model=./vits-ljs.onnx \
  --vits-lexicon=./lexicon.txt \
  --vits-tokens=./tokens.txt \
  --output-filename=./tts/vits-ljs.wav \
  'liliana, the most beautiful and lovely assistant of our team!'

ls -lh ./tts

rm -v vits-ljs.onnx ./lexicon.txt ./tokens.txt

log "vits-vctk test"
wget -qq https://huggingface.co/csukuangfj/vits-vctk/resolve/main/vits-vctk.onnx
wget -qq https://huggingface.co/csukuangfj/vits-vctk/resolve/main/lexicon.txt
wget -qq https://huggingface.co/csukuangfj/vits-vctk/resolve/main/tokens.txt

for sid in 0 10 90; do
  $EXE \
    --vits-model=./vits-vctk.onnx \
    --vits-lexicon=./lexicon.txt \
    --vits-tokens=./tokens.txt \
    --sid=$sid \
    --output-filename=./tts/vits-vctk-${sid}.wav \
    'liliana, the most beautiful and lovely assistant of our team!'
done

rm -v vits-vctk.onnx ./lexicon.txt ./tokens.txt
ls -lh tts/

log "vits-zh-aishell3"

wget -qq https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/vits-aishell3.onnx
wget -qq https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/lexicon.txt
wget -qq https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/tokens.txt

for sid in 0 10 90; do
  $EXE \
    --vits-model=./vits-aishell3.onnx \
    --vits-lexicon=./lexicon.txt \
    --vits-tokens=./tokens.txt \
    --sid=$sid \
    --output-filename=./tts/vits-aishell3-${sid}.wav \
    '林美丽最美丽'
done

rm -v vits-aishell3.onnx ./lexicon.txt ./tokens.txt

ls -lh ./tts/
