#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "CXX_STREAMING_ZIPFORMER_EXE is $CXX_STREAMING_ZIPFORMER_EXE"
echo "PATH: $PATH"

log "------------------------------------------------------------"
log "Test streaming zipformer CXX API"
log "------------------------------------------------------------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
$CXX_STREAMING_ZIPFORMER_EXE
rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
