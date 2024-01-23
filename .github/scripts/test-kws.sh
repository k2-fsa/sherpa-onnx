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
log "Run Chinese keyword spotting (Wenetspeech）"
log "------------------------------------------------------------"

repo_url=https://github.com/pkufool/keyword-spotting-models/releases/download/v0.1/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz
log "Start testing ${repo_url}"
repo=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01

log "Download pretrained model and test-data from $repo_url"
wget $repo_url
tar jxvf ${repo}.tar.bz

time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
  --decoder=$repo/decoder-epoch-12-avg-2-chunk-16-left-64.onnx \
  --joiner=$repo/joiner-epoch-12-avg-2-chunk-16-left-64.onnx \
  --keywords-file=$repo/test_wavs/test_keywords.txt \
  --max-active-paths=4 \
  --num-threads=4 \
  $repo/test_wavs/3.wav $repo/test_wavs/4.wav $repo/test_wavs/5.wav $repo/test_wavs/6.wav

rm -rf $repo
rm -rf ${repo}.tar.bz

log "------------------------------------------------------------"
log "Run English keyword spotting (Gigaspeech）"
log "------------------------------------------------------------"

repo_url=https://github.com/pkufool/keyword-spotting-models/releases/download/v0.1/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz
log "Start testing ${repo_url}"
repo=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01

log "Download pretrained model and test-data from $repo_url"
wget $repo_url
tar jxvf ${ropo}.tar.bz

time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
  --decoder=$repo/decoder-epoch-12-avg-2-chunk-16-left-64.onnx \
  --joiner=$repo/joiner-epoch-12-avg-2-chunk-16-left-64.onnx \
  --keywords-file=$repo/test_wavs/test_keywords.txt \
  --max-active-paths=4 \
  --num-threads=4 \
  $repo/test_wavs/0.wav $repo/test_wavs/1.wav

rm -rf $repo
rm -rf ${repo}.tar.bz
