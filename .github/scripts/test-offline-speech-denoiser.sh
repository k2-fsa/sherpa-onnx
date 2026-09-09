#!/usr/bin/env bash

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z $EXE ]; then
  EXE=./build/bin/sherpa-onnx-offline-denoiser
fi

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run gtcrn"
log "------------------------------------------------------------"
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/speech_with_noise.wav

$EXE \
  --debug=1 \
  --speech-denoiser-gtcrn-model=./gtcrn_simple.onnx \
  --input-wav=./speech_with_noise.wav \
  --output-wav=./enhanced_speech_16k.wav

rm ./gtcrn_simple.onnx
