#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "SLID_EXE is $SLID_EXE"
echo "PATH: $PATH"


log "------------------------------------------------------------"
log "Download whisper tiny for spoken language identification    "
log "------------------------------------------------------------"

rm -rf sherpa-onnx-whisper-tiny*
curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
rm sherpa-onnx-whisper-tiny.tar.bz2

$SLID_EXE

rm -rf sherpa-onnx-whisper-tiny*
