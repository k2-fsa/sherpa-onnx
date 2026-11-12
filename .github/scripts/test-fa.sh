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

download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/zh.wav

# Models: https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
for model in zh-aishell3-pinyin zh-biaobei-pinyin zh-large-pinyin; do
  log "Test $model"
  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/${model}.tar.bz2

  ls -lh $model

  sherpa-onnx-fa \
    --model-dir=$model \
    --lexicon=$model/lexicon.txt \
    --wav=zh.wav \
    --text="不要问你的国家能为你做什么，而要问你能为你的国家做什么。" \
    --out=${model}.TextGrid

  ls -lh ${model}.TextGrid
  grep -q 'name = "phones"' ${model}.TextGrid
  grep -q 'name = "words"' ${model}.TextGrid

  rm -rf $model
done

log "generated TextGrid files:"
ls -lh *.TextGrid
