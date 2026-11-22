#!/usr/bin/env bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "------------------------------------------------------------"
log "Test sherpa-onnx-fa (forced alignment)                      "
log "------------------------------------------------------------"

# Models: https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
# Each model package ships its own test wav (zh.wav / en.wav) whose
# transcript is documented in the README.md of the package.
# Pass the transcripts via --text-file: non-ASCII values given with --text
# are mangled on Windows, where the CRT converts the command line to the
# ANSI code page before main() sees argv (Chinese becomes '???').
printf '%s' "不要问你的国家能为你做什么，而要问你能为你的国家做什么。" > zh-text.txt
printf '%s' "ask not what your country can do for you, ask what you can do for your country." > en-text.txt

for model in english-mfa-v2.2.1 english-mfa-v3.0.0 english-mfa-v3.1.0; do
  log "Test $model"
  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/${model}.tar.bz2

  ls -lh $model

  sherpa-onnx-fa \
    --model-dir=$model \
    --lexicon=$model/lexicon.txt \
    --wav=$model/en.wav \
    --text-file=en-text.txt \
    --out=${model}.TextGrid

  ls -lh ${model}.TextGrid
  grep -q 'name = "phones"' ${model}.TextGrid
  grep -q 'name = "words"' ${model}.TextGrid

  cp $model/en.wav ./en.wav  # keep the wav for the artifact upload
  rm -rf $model
done

for model in zh-aishell3-pinyin zh-biaobei-pinyin zh-large-pinyin; do
  log "Test $model"
  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/${model}.tar.bz2

  ls -lh $model

  sherpa-onnx-fa \
    --model-dir=$model \
    --lexicon=$model/lexicon.txt \
    --wav=$model/zh.wav \
    --text-file=zh-text.txt \
    --out=${model}.TextGrid

  ls -lh ${model}.TextGrid
  grep -q 'name = "phones"' ${model}.TextGrid
  grep -q 'name = "words"' ${model}.TextGrid

  cp $model/zh.wav ./zh.wav  # keep the wav for the artifact upload
  rm -rf $model
done


log "generated TextGrid files:"
ls -lh *.TextGrid
